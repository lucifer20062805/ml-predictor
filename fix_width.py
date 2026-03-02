import os

def fix_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Revert width="stretch" back to use_container_width=True
    content = content.replace('width="stretch"', 'use_container_width=True')
    content = content.replace("width='stretch'", "use_container_width=True")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

fix_file('dashboard.py')
fix_file('app.py')
print("Successfully reverted width attribute!")
