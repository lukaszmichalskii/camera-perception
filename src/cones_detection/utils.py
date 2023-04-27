def get_class_list(class_file):
    classes = []
    with open(class_file) as fd:
        class_ = fd.readline().strip()
        while class_:
            ids, cone = class_.split(' ')
            classes.append((ids, cone))
            class_ = fd.readline().strip()
    return classes


def cuda_tensor_to_cpu(tensor):
    return tensor.cpu()
