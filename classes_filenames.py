import pickle
from config import root_dir
from model_picker import generator

filenames = [root_dir + '/' + s for s in generator.filenames]
print(len(list(generator.classes)))
print(len(filenames))

pickle.dump(generator.classes, open('class_ids-caltech101.pickle', 'wb'))
pickle.dump(filenames, open('filenames-caltech101.pickle', 'wb'))
