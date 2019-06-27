Date=$(date +%Y%m%d%H%M)
TAG="Release_$Date"

echo "#### 开始发布 ####"

echo "Relase OCR Web程序..."
git add .
git add -u
git commit -m"$TAG"
git tag $TAG
git push --tags

echo "Relase CTPN 程序..."
cd ../ctpn
git add .
git add -u
git commit -m"$TAG"
git tag $TAG
git push --tags

echo "Relase CRNN 程序..."
cd ../crnn
git add .
git add -u
git commit -m"$TAG"
git tag $TAG
git push --tags

cd ../ocr
echo "发布完毕..."
