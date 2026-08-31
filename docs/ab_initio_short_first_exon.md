# Ab initio 避免錯誤的短 first CDS exon + 長 intron

## 問題界定

這個錯誤至少有兩種，不能只用同一條 Kozak 規則處理：

1. **TIS 錯了，但 splice structure 對**：第一個短 CDS segment 應該是 5′ UTR，真正的 ATG 在下游 exon。`fix_orf.py` 的 Kozak switch 能處理一部分這類情況，但它要求同 frame、同 stop，並保留原有 intron。
2. **第一個 exon／donor／intron 本身是假的**：此時 post-hoc 移動 ATG 不夠，因為應該讓「下游開始的新 gene path」或其他 splice path 在全域解碼中勝出。

GeneCAD 已有很好的起點：`FrameStateGraph` 強制合法 ATG、stop、reading frame 和 canonical splice motifs，並用
`(length / 9) ** strictness` 對短 coding run 加 soft penalty；repo 內的實測也顯示 hard 9-nt floor 會傷害真實短 exon。現在的限制是：

- decoder 仍以五類 per-base emission 加 feature transition 做單一 Viterbi path；
- splice boundary 只要求 dinucleotide 合法，沒有在 transition 上直接比較「這個 donor/acceptor 有多像真的」；
- intron body loop 隱含近似 geometric duration，短 exon prior 只覆蓋 `<9 nt`；
- `include_utr_in_coding_run=True` 把 physical first exon 長度和 first **CDS** segment 長度混成一個量；
- Kozak 在 `fix_orf.py` 才加入，無法和另一條 gene/splice path 直接競爭。

因此目標不應是「禁止短 exon」或「禁止長 intron」，而是讓短 exon 只有在 start、splice、coding、duration 和整條 ORF 的聯合證據都夠強時才保留。

## 有文獻支持的 intrinsic 方法

### 1. Explicit-duration／semi-Markov length model

GENSCAN 的 generalized HMM 同時建模 transcription/translation/splice signals、exon/intron composition 與 length distributions；GeneMark-ES 的 HSMM 更明確為 initial、internal、terminal、single-exon coding states 建立三週期 coding model及由資料平滑得到的 state-duration distributions。這說明 **initial CDS exon 應是獨立 state/type**，不應只共享一個通用 CDS→intron transition。[Burge & Karlin 1997](https://pubmed.ncbi.nlm.nih.gov/9149143/)，[Lomsadze et al. 2005](https://academic.oup.com/nar/article/33/20/6494/1082033)

AUGUSTUS 的重要警告是：普通 HMM self-loop 只能產生 shifted-geometric intron length；它對短 intron 的形狀不準，也會把真正的長 intron 罰得過重。AUGUSTUS 對較短部分使用 empirical explicit distribution，長尾才接 geometric state。[Stanke & Waack 2003](https://gobics.de/mario/papers/GenePred2003.pdf)

對 GeneCAD 的含義：

- 對 `initial CDS length` 建 empirical/smoothed log prior，至少精細覆蓋容易出錯的 3–60 nt；超過 cap 後進 tail state。
- intron duration 也可用「empirical head + geometric tail」，但**不能把 long intron 一律加重罰**；那會拆掉真正含長 intron 的 plant genes。
- 可評估一個正則化的 interaction `log P(L_initial, L_intron)` 或 `log P(L_initial | intron-bin)`，但只有在 held-out data 確認短+長組合確實異常時才採用。稀疏 2D counts 必須 shrink 到兩個 marginal priors，避免把罕見真基因學成零機率。

### 2. 對 splice site 評分，而不只檢查 GT/GC–AG

GeneID 先用 log-likelihood-ratio PWM 分別評 start、donor、acceptor，再把 defining-site scores 和 frame-specific coding log-likelihood 相加成 exon score，最後用 dynamic programming 找最高分 gene structure。[Parra, Blanco & Guigó 2000](https://genome.crg.es/courses/Lisbon04/papers/paper3.pdf)

Plant-specific 結果也支持使用 dinucleotide 以外的 context：SplicePredictor 在 maize／Arabidopsis 使用 splice-site sequence quality，以及 junction 兩側 U 和 GC composition contrast；高分群具有較高 specificity。[Brendel et al. 1998](https://academic.oup.com/nar/article/26/20/4748/2902399)

對 GeneCAD 最直接的改進是，在進入 donor chain 和離開 acceptor chain 的 edge 上加入 calibrated local LLR（或一個小型 plant donor/acceptor classifier 的 logit）。真正的 3–8 nt initial CDS exon 幾乎沒有足夠長度展示 coding periodicity，所以強 start + 強 donor + 強 acceptor 對保留它尤其重要。

### 3. Kozak／TIS 應進入 joint decoding，但不能單獨裁決

植物 AUG context 不是通用常數：5,074 個 plant genes 的分析發現 −3、+4 常見 purine，且 monocot 和 dicot consensus 有差異。[Joshi et al. 1997](https://pubmed.ncbi.nlm.nih.gov/9426620/)

因此可把 `kozak_score()` 變成 `5′UTR -> start_a` edge 的一項分數，讓以下兩條完整路徑直接競爭：

```text
path A: upstream ATG -> 3–8 nt CDS -> long intron -> downstream CDS -> same stop
path B: upstream UTR/intergenic -> downstream ATG -> downstream CDS -> same stop
```

建議比較的是完整 path 分數，而不是 `alternative_kozak - original_kozak > 3`：

```text
S(path) = neural emission + grammar transition
        + w_start  * TIS-context score
        + w_splice * (donor + acceptor scores)
        + duration priors
        + w_coding * frame-specific coding LLR
```

這也涵蓋 whole-ORF consistency：ATG、stop、無 internal stop、跨 intron phase、initial segment 與後續 CDS 的三週期 coding potential 一起決定。AUGUSTUS 本身也聯合 translation-initiation motif、initial-exon content、splice models、coding content 和 duration，而不是先選 exon 再用 start motif 獨立翻案。[Stanke & Waack 2003](https://gobics.de/mario/papers/GenePred2003.pdf)

### 4. 額外的 coding periodicity/content score

GeneID 對 coding 和 intron 分別訓練 order-5 Markov models，coding model依三個 codon positions 分開，將 frame-specific coding/noncoding LLR 納入 exon score。[Parra et al. 2000](https://genome.crg.es/courses/Lisbon04/papers/paper3.pdf)

GeneCAD 的 transformer emissions 很可能已學到這項訊號，但短 initial segment 會因樣本太少而不穩。可先把 explicit coding LLR 當 diagnostic feature；若 held-out ablation 顯示有增益，再以可學權重加入 path score。不要假設 raw neural log-probability、Kozak log2-odds 和 Markov natural-log LLR 在同一尺度。

### 5. Posterior／n-best：證據不足時保留不確定性

現在 `_masked_viterbi()` 只回傳最佳 path。AUGUSTUS 會輸出 exon、intron、transcript 的 posterior probability；其 exon posterior 取決於相容的鄰近 exons，而不只是該 exon 本身，並可由 sampling 產生 alternative transcripts。其文件也明確警告 model posterior 可能過度自信，需再校準。[AUGUSTUS official README](https://github.com/nextgenusfs/augustus/blob/master/README.TXT#L2783-L2886)

GeneCAD 可先做成本較低的 constrained two-best diagnostic：

- `S_short`：強制包含可疑 short-first-CDS boundary 的最佳 path；
- `S_alt`：禁止該 boundary（或強制下游 start）的最佳 path；
- 用 `Δ = S_short - S_alt` 決定 accept／repair／flag，而不是只看 top-1 label。

之後再實作 sparse forward-backward 或 n-best。對 intrinsic sequence 無法辨識的 3-nt exon，誠實的 `ambiguous_short_first_exon` 比硬改更能保持 sensitivity。

### 6. Species/genome calibration

SNAP 顯示 gene prediction 對 species-specific parameters 敏感，最近的親緣物種也不一定提供最相容參數；GeneMark-ES 則用受限制的 iterative Viterbi training，僅從 anonymous genomic DNA 自訓 coding、noncoding、site 和 duration parameters，並在 Arabidopsis 等測試。[Korf 2004](https://link.springer.com/article/10.1186/1471-2105-5-59)，[Lomsadze et al. 2005](https://academic.oup.com/nar/article/33/20/6494/1082033)

可延伸 GeneCAD 現有 per-genome Kozak-margin calibration，但「GeneCAD 自己預測的長 first exon」不是 known-correct truth，屬 pseudo-label。較安全做法是：

- 以高 posterior／大 two-best margin、完整 ORF、強雙 splice sites 的 predictions 迭代估計；
- 對 length/site/start priors 設 clade prior 與 shrinkage，防止 self-training drift；
- 所有 thresholds/weights 在 held-out **species** 上校準，不只 random genes，避免同物種 leakage。

## GeneCAD 建議設計與順序

1. **先加 observability，不改 prediction**：輸出每個可疑 locus 的原/替代 TIS、Kozak、donor、acceptor、first-CDS length、physical-first-exon length、intron length、coding LLR、top-2 path gap。這會先回答錯的是 TIS 還是 splice path。
2. **把兩種長度拆開**：保留 `physical first exon = 5′UTR + CDS` 的 splicing prior，同時另設 `first CDS segment` prior。不要讓長 5′ UTR 自動取消所有 short-CDS 警示。
3. **加入 boundary scores**：以 plant/clade-specific donor/acceptor local LLR 加權 edge；先在 held-out species 做 ablation。
4. **把 TIS score 移入 frame decoder**：保留 `fix_orf` 作 validation/safety net，但讓 upstream-short-exon path 和 downstream-start path在同一 objective 競爭。
5. **以 capped semi-Markov duration 取代手工 power penalty**：先只做 initial CDS empirical head；確認收益後才改 intron head/tail。若 2D `short initial × long intron` feature 沒有跨物種穩定增益，就不要加入。
6. **最後加 uncertainty policy**：高 `Δ` 才自動選；中間區間 flag/保留 n-best；低 `Δ` 保留原預測。校準目標同時報 short-first-exon precision/recall、exact-locus F1、TIS accuracy，並按 first-CDS 與 intron-length bins 分層。

這條路徑與現有 `FrameStateGraph` 相容：forced chains 已證明 state expansion 可行；可用 capped duration states 延伸，而不必推翻 neural encoder。現代 Helixer 也採「sequence-only deep model + structured HMM」產生完整 gene model，支持這種組合架構。[Holst et al. 2025](https://www.nature.com/articles/s41592-025-02939-1)

## 主要 pitfalls

- **硬 minimum exon length**：會直接刪掉真實 3–8 nt exon；repo 自己的 chromosome-level 結果已觀察到 locus F1 下降。
- **long intron 一律重罰**：會拆 gene；AUGUSTUS 的 empirical-head/geometric-tail 正是為修正這個問題。
- **只用 Kozak**：弱 context 的真 start、leaky scanning、monocot/dicot差異都會造成錯誤；Kozak 應是 joint evidence。
- **重複計分**：transformer emission 已由同一 DNA context 得出；額外 PWM/Markov score 可能 double-count，權重必須用 held-out calibration 學習。
- **把 canonical motif 當真 splice site**：GT…AG 常可偶然出現；需要完整 junction context 與 competing paths。
- **把 pseudo-label 當 truth**：現有 per-genome calibration 的「confident starts」仍是模型預測。自訓需限制更新、shrinkage 和停止條件。
- **只看整體 F1**：這個 failure mode 很稀少，必須另外追蹤 `<9 nt`、intron length deciles、monocot/dicot、UTR-present/absent strata。

## 結論

最值得先做的不是另一個 hard filter，而是 **plant-specific splice boundary scores + initial-CDS duration prior + Kozak/ORF joint path comparison + calibrated two-best uncertainty**。這四者能壓掉弱證據的假 short-first-exon path，同時讓擁有強 start、強 splice 與一致 coding evidence 的真短 exon 留下。單純提高 `exon_length_strictness` 或懲罰 long intron，無法可靠達成這個 precision/sensitivity 平衡。
