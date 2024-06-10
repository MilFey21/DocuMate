import os
import pandas as pd

from modules.extract_embedding import text_to_embed


def get_files_in_folder(folder_path):
    allowed_formats = ['.txt', '.docx', '.epub', '.pdf', '.csv', '.xls', '.xlsx', '.ppt', '.html']
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in allowed_formats]
    return files


def get_target_folders(root_folder: str, config) -> pd.DataFrame:
    folder_data = []
    for folder, _, files in os.walk(root_folder):
        dmate_file = [f for f in files if f.endswith('.dmate')]
        if dmate_file:
            dmate_file_path = os.path.join(folder, dmate_file[0])
            with open(dmate_file_path, 'r') as file:
                dmate_content = file.read()
                folder_data.append(
                    {
                        'folder_path': folder,
                        'content': dmate_content,
                        'content_emb': text_to_embed(dmate_content, config)
                    }
                )
    return pd.DataFrame(folder_data)