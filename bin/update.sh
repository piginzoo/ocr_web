if [ "$1" == "" ]; then
    echo "必须提供Realse Tag名字"
    exit;
fi

echo "#### 使用Release版本：$1 ####"

echo "更新 OCR Web程序..."
git reset --hard
git pull
git checkout $1

echo "更新 CTPN 程序..."
cd ../ctpn
git reset --hard
git pull
git checkout $1

echo "更新 CRNN 程序..."
cd ../crnn
git reset --hard
git pull
git checkout $1

cd ../ocr
echo "更新完毕..."
echo "已经切换到版本：$1!"