import os
import glob
import subprocess

current_workdir = os.getcwd()
download_files = glob.glob('./**/download.sh', recursive=True)
for file in download_files:
    if 'skin' in file:
        os.chdir(file[:-len('download.sh')])
        subprocess.call('./download.sh')
        os.chdir(current_workdir)
