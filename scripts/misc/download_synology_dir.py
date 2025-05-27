#!/usr/bin/env python3

"""
This script generates wget commands for downloading files from Synology Drive.

Usage:
    ./download_synology_dir.py <json_file>
    cat <json_file> | ./download_synology_dir.py

This json string can be copied from the synology UI via developer tools after 
clicking on a directory and seeing an API response like the following:

{
    "data": {
        "items": [
            {
                "access_time": 1747438168,
                "adv_shared": false,
                "app_properties": {
                    "type": "none"
                },
                "capabilities": {
                    "can_comment": false,
                    "can_delete": false,
                    "can_download": true,
                    "can_encrypt": false,
                    "can_organize": false,
                    "can_preview": true,
                    "can_read": true,
                    "can_rename": false,
                    "can_share": false,
                    "can_sync": true,
                    "can_write": false
                },
                "change_id": 262520,
                "change_time": 1747432650,
                "content_snippet": "",
                "content_type": "document",
                "created_time": 1747432650,
                "disable_download": false,
                "display_path": "/shared-with-me/data_share/training_data/new_data/Acoerulea_322_v3_processing.log",
                "dsm_path": "",
                "enable_watermark": false,
                "encrypted": false,
                "file_id": "885116949514729241",
                "force_watermark_download": false,
                "hash": "8098a8ef967d4a25b06b8e89e663e636",
                "image_metadata": {
                    "time": 1747432649
                },
                "labels": [],
                "max_id": 262520,
                "modified_time": 1747432649,
                "name": "Acoerulea_322_v3_processing.log",
                "owner": {
                    "display_name": "zongyan",
                    "name": "zongyan",
                    "nickname": "",
                    "uid": 1026
                },
                "parent_id": "885116707478222573",
                "path": "/Abroad/Cornell University/Research/BitBucket/gene_annotation/openathena/data_share/training_data/new_data/Acoerulea_322_v3_processing.log",
                "permanent_link": "13NqOFjTXlgNrWDnQjaTflGNuAmI3maq",
                "properties": {},
                "removed": false,
                "revisions": 1,
                "shared": false,
                "shared_with": [],
                "size": 14906,
                "starred": false,
                "support_remote": false,
                "sync_id": 262520,
                "sync_to_device": false,
                "transient": false,
                "type": "file",
                "version_id": "262520",
                "watermark_version": 0
            },
            ...
        ],
        "total": 23
    },
    "success": true
}
"""

import json
import sys

def generate_wget_commands(json_string):
    """Generate wget commands for each file in the JSON data."""
    try:
        # Parse the JSON string
        data = json.loads(json_string)
        
        # Loop through each item in the data
        for item in data.get('data', {}).get('items', []):
            filename = item.get('name')
            file_id = item.get('file_id')
            
            if not filename or not file_id:
                continue
                
            # Generate the wget command
            wget_cmd = (
                f"wget -O {filename} 'https://zy-liu.synology.me:4249/d/s/12cQtnvzAo5AxeGjcpNXSeDquYgRtHXX/webapi/entry.cgi/SYNO.SynologyDrive.Files/{filename}"
                f"?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A{file_id}%22%5D&force_download=true&json_error=true"
                f"&download_type=%22download%22&c2_offload=%22allow%22&_dc=1747407662957&sharing_token=%22u92ubJbkJy4bC0EOeSYOgTYAEdveuzDL8I4gUHxp."
                f"YRfTGWmP4gfdUOCZ9eKfp15xVtpnpoUeBGohc021dz_NPUX_TlBAsrGoJI6fTLPD7Kp79NgWDF7O0b1jeTnFbF5nOWd58GjHyB2R2uhgoogjCC0N6OrW3."
                f"FA0A1LdozbEsCKQTayFkKyY0aaQyBuEzr970RTduK7yuYkh.f7BJFAWywYTDwbltdIkahhAFi.2AtHljeAJJW2gU-%22'"
            )
            
            print(wget_cmd)
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    # If JSON is provided as input file
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as f:
                json_string = f.read()
            generate_wget_commands(json_string)
        except FileNotFoundError:
            print(f"File not found: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    # Otherwise read from stdin
    else:
        json_string = sys.stdin.read()
        sys.exit(generate_wget_commands(json_string))



