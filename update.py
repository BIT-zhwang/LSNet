import os
import zipfile
import time

def fileFolderCreate():
    pass

def fileCompress(srcDir,desDir,zipName):
    filelist=[]
    for root,dirs,files in os.walk(srcDir):
        for name in files:
            filelist.append(os.path.join(root,name))
    zf=zipfile.ZipFile(zipName,'w',zipfile.ZIP_DEFLATED)

    for file in filelist:
        print(file)
        zf.write(file)
        time.sleep(0.1)
    zf.close()

    time.sleep(1)

    print("Compressed file success.")

if __name__=='__main__':
    updateFolderDir=""
    buildFolderDir=""

    fileCompress('Documents','','docu')
