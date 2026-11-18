import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

def patch_notebook(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # comment out shell commands and colab file downloads
                if line.lstrip().startswith('!'):
                    line = '# ' + line
                if 'files.upload()' in line or 'shutil.copy(\'kaggle.json\'' in line or 'files.download(' in line or 'google.colab' in line:
                    line = '# ' + line
                # change paths from /content/ to local workspace
                if '/content/' in line:
                    line = line.replace('/content/', f'{BASE_DIR}/')
                
                new_source.append(line)
            cell['source'] = new_source
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
patch_notebook(os.path.join(BASE_DIR, 'notebooks/01_train_skin_model.ipynb'))
patch_notebook(os.path.join(BASE_DIR, 'notebooks/02_build_ingredient_db.ipynb'))
print("Notebooks patched.")
