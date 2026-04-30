# 1. 创建一个专门存放数据的文件夹（推荐），并进入该文件夹
mkdir -p data/coco
cd data/coco

# 2. 使用 wget 下载验证集压缩包 (文件大小约 770MB)
wget http://images.cocodataset.org/zips/val2017.zip

# 3. 解压下载的 zip 文件 (如果提示找不到 unzip 命令，请先运行 sudo apt install unzip)
unzip val2017.zip

# 4. (可选) 解压完成后删除压缩包以节省硬盘空间
rm val2017.zip