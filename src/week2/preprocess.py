import numpy as np
import graphlab
from scipy.sparse import csr_matrix

def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    return(norm)

def sframe_to_scipy(column):
    # Create triples of (row_id, feature_id, count).
    x_frame = graphlab.SFrame({'X1':column})
    
    # 1. Add a row number.
    x_frame_row = x_frame.add_row_number()
    del x_frame
    # 2. Stack will transform x to have a row for each unique (row, key) pair.
    x_stack = x_frame_row.stack('X1', ['feature', 'value'])
    del x_frame_row

    # Map words into integers using a OneHotEncoder feature transformation.
    f = graphlab.feature_engineering.OneHotEncoder(features=['feature'])

    # We first fit the transformer using the above data.
    f.fit(x_stack)

    # The transform method will add a new column that is the transformed version
    # of the 'word' column.
    x = f.transform(x_stack)
    del x_stack

    # Get the feature mapping.
    mapping = f['feature_encoding']

    # Get the actual word id.
    x['feature_id'] = x['encoded_features'].dict_keys().apply(lambda x: x[0])

    # Create numpy arrays that contain the data for the sparse matrix.
    i = np.array(x['id'])
    j = np.array(x['feature_id'])
    v = np.array(x['value'])
    width = x['id'].max() + 1
    height = x['feature_id'].max() + 1

    # Create a sparse matrix.
    mat = csr_matrix((v, (i, j)), shape=(width, height))

    del x
    del v, i, j

    return mat, mapping
