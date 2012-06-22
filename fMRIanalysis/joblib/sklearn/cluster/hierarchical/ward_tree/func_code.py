# first line: 30
def ward_tree(X, connectivity=None, n_components=None, copy=True):
    """Ward clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account a some topological
    structure between samples.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered

    connectivity : sparse matrix.
        connectivity matrix. Defines for each sample the neigbhoring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_components : int (optional)
        Number of connected components. If None the number of connected
        components is estimated from the connectivity matrix.

    copy : bool (optional)
        Make a copy of connectivity or work inplace. If connectivity
        is not of LIL type there will be a copy in any case.

    Returns
    -------
    children : list of pairs. Lenght of n_nodes
               list of the children of each nodes.
               Leaves of the tree have empty list of children.

    n_components : sparse matrix.
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))

    if connectivity is None:
        out = hierarchy.ward(X)
        children_ = out[:, :2].astype(np.int)
        return children_, 1, n_samples

    # Compute the number of nodes
    if n_components is None:
        n_components, labels = cs_graph_components(connectivity)

    # Convert connectivity matrix to LIL with a copy if needed
    if sparse.isspmatrix_lil(connectivity) and copy:
        connectivity = connectivity.copy()
    else:
        connectivity = connectivity.tolil()

    if n_components > 1:
        warnings.warn("the number of connected components of the"
        " connectivity matrix is %d > 1. Completing it to avoid"
        " stopping the tree early."
        % n_components)
        connectivity = _fix_connectivity(X, connectivity,
                                            n_components, labels)
        n_components = 1

    n_nodes = 2 * n_samples - n_components

    if (connectivity.shape[0] != n_samples or
        connectivity.shape[1] != n_samples):
        raise ValueError('Wrong shape for connectivity matrix: %s '
                         'when X is %s' % (connectivity.shape, X.shape))

    # Remove diagonal from connectivity matrix
    connectivity.setdiag(np.zeros(connectivity.shape[0]))

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.int)
    coord_col = np.array(coord_col, dtype=np.int)

    # build moments as a list
    moments_1 = np.zeros(n_nodes)
    moments_1[:n_samples] = 1
    moments_2 = np.zeros((n_nodes, n_features))
    moments_2[:n_samples] = X
    inertia = np.empty(len(coord_row), dtype=np.float)
    _hierarchical.compute_ward_dist(moments_1, moments_2,
                             coord_row, coord_col, inertia)
    inertia = zip(inertia, coord_row, coord_col)
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.int)
    heights = np.zeros(n_nodes)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []

    visited = np.empty(n_nodes, dtype=bool)

    # recursive merge loop
    for k in xrange(n_samples, n_nodes):
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j], heights[k] = k, k, inert
        children.append([i, j])
        used_node[i] = used_node[j] = False

        # update the moments
        moments_1[k] = moments_1[i] + moments_1[j]
        moments_2[k] = moments_2[i] + moments_2[j]

        # update the structure matrix A and the inertia matrix
        coord_col = []
        visited[:] = False
        visited[k] = True
        for l in set(A[i]).union(A[j]):
            l = _hierarchical._get_parent(l, parent)
            if not visited[l]:
                visited[l] = True
                coord_col.append(l)
                A[l].append(k)
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.int)
        coord_row = np.empty_like(coord_col)
        coord_row.fill(k)
        ini = np.empty(len(coord_row), dtype=np.float)

        _hierarchical.compute_ward_dist(moments_1, moments_2,
                                   coord_row, coord_col, ini)
        for tupl in itertools.izip(ini, coord_row, coord_col):
            heappush(inertia, tupl)

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    children = np.array(children)  # return numpy array for efficient caching

    return children, n_components, n_leaves
