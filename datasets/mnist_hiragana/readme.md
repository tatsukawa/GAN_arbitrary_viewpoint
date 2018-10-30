# Hiragana Images

You need to download the dataset from [here](https://github.com/inoueMashuu/hiragana-dataset).

## How to use
```python
import loader
path = ''
images = loader.read(path, 10) # You can get 10 images,
images.shape # (10, 1, 28, 28)
```