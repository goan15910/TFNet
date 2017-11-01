# Dataset module
We deploy a plug-in-fashion implementation to support future dataset.
To implement your own dataset, you need to:

1. Inherit class ```Dataset``` in dataset.py

2. Define a easydict called ```self.fname_dict```, which defines the mapping from ```SET``` key (```SET.TRAIN```) to filename ('train.txt')

3. Implement a member funcion, following the name & format:
```python
   def _read_fnames(self, filename):
     # YOUR IMPLEMENTATION
     # READ IN THE FILE AND APPEND ALL DATA/LABEL FILENAME INTO THE LIST 'fnames'
     return fnames
```

4. Implement a member function, following the name & format:
```python
   def _decode_func(self, idxs):
     def my_decode_func(idxs):
       # YOUR IMPLEMENTATION
       return batch
     return my_decode_func
```

5. Register your module
