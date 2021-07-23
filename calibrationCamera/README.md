# 标定相机
利用棋盘阵拍摄大概二十张图片，保存到一个文件夹中[imagepath]

# 标定
编译以后使用方法：
```shell
./calibrationCamera [imagepath] [width-1] [height-1] [squareSize = 1.0]\n
```
width就是棋盘格子列数；height就是棋盘格式行数,squareSize就是黑白格子的边长，单位；
比如10*10的棋盘；输入
```shell
./calibrationCamera [imagepath] 9 9 20
```
