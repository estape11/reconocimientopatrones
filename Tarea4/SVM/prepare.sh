gzip -d -k train-images-idx3-ubyte.gz
gzip -d -k train-labels-idx1-ubyte.gz
gzip -d -k t10k-images-idx3-ubyte.gz
gzip -d -k t10k-labels-idx1-ubyte.gz

mkdir -p samples
mv train-images-idx3-ubyte samples
mv train-labels-idx1-ubyte samples
mv t10k-images-idx3-ubyte samples
mv t10k-labels-idx1-ubyte samples