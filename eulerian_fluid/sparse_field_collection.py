from sparse_field import SparseField


class SparseFieldCollection:
    def __init__(self, num_copies, **kwargs):
        self.sparse_fields = []
        for i in range(num_copies):
            self.sparse_fields.append(SparseField(**kwargs))

    def __getitem__(self, item):
        return self.sparse_fields[item]

    def __len__(self):
        return len(self.sparse_fields)

    def swap(self, a, b):
        self.sparse_fields[a], self.sparse_fields[b] = self.sparse_fields[
            b], self.sparse_fields[a]
