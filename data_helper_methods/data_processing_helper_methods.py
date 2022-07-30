import torchvision.transforms as transforms
import os
from custom_datasets.CustomFlowerDataset import CustomFlowerDataSet
from custom_datasets.CustomCubDataSet import CustomCubDataSet
from custom_datasets.CustomFoodDataSet import CustomFoodDataSet
from custom_datasets.CustomCaltechDataset import CustomCaltechDataSet
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

BATCH_SIZE = 256
NUM_WORKERS = 0

class DataProcessingHelperMethods:

    @staticmethod
    def prepare_flowers_dataset(base_path):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        training_path = base_path + "train/"
        valid_path = base_path + "valid/"
        test_path = base_path + "test/"

        training_folders = os.listdir(training_path)
        total_classes = len(training_folders)
        training_files = []
        training_labels = []
        for folder in training_folders:
            label = int(folder) - 1
            files = os.listdir(training_path + folder + "/")
            files = [training_path + folder + "/" + fil for fil in files]
            training_files.extend(files)
            training_labels.extend([label] * len(os.listdir(training_path + folder + "/")))

        valid_folders = os.listdir(valid_path)
        valid_files = []
        valid_labels = []
        for folder in valid_folders:
            label = int(folder) - 1
            files = os.listdir(valid_path + folder + "/")
            files = [valid_path + folder + "/" + fil for fil in files]
            valid_files.extend(files)
            valid_labels.extend([label] * len(os.listdir(valid_path + folder + "/")))

        test_files = os.listdir(test_path)
        test_labels = [0] * len(test_files)

        train_dataset = CustomFlowerDataSet(transform, training_files, training_labels)
        val_dataset = CustomFlowerDataSet(transform, valid_files, valid_labels)
        test_dataset = CustomFlowerDataSet(transform, test_files, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

        return train_dataloader, valid_dataloader, test_dataloader, total_classes

    @staticmethod
    def prepare_cub_dataset(base_path):
        train_test_split = {}
        id_to_label = {}
        id_to_image_name = {}

        with open(base_path + "train_test_split.txt", "r") as f:
            readlines = f.readlines()

        for line in readlines:
            line = line.split("\n")[0]
            id_, label = line.split(" ")
            train_test_split[id_] = int(label)

        with open(base_path + "image_class_labels.txt", "r") as f:
            readlines = f.readlines()

        for line in readlines:
            id_, label = line.split("\n")[0].split(" ")
            id_to_label[id_] = int(label) - 1

        with open(base_path + "images.txt", "r") as f:
            readlines = f.readlines()

        for line in readlines:
            id_, name = line.split("\n")[0].split(" ")
            id_to_image_name[id_] = name

        total_classes = len(np.unique(list(id_to_label.values())))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # get values in training and test dataset

        counter = defaultdict(int)
        for key in train_test_split.keys():
            counter[train_test_split[key]] += 1
        print("count for training and test dataset ", counter)

        # Make Dataloaders for train and test split

        training_files = []
        training_labels = []
        test_files = []
        test_labels = []
        for key in train_test_split.keys():
            if train_test_split[key] == 0:
                training_files.append(id_to_image_name[key])
                training_labels.append(id_to_label[key])
            else:
                test_files.append(id_to_image_name[key])
                test_labels.append(id_to_label[key])
        train_dataset = CustomCubDataSet(base_path + "images/", transform, training_files, training_labels)
        test_dataset = CustomCubDataSet(base_path + "images/", transform, test_files, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        return train_dataloader, test_dataloader, test_dataloader, total_classes

    @staticmethod
    def prepare_food_dataset(base_path):
        meta_path = base_path

        with open(meta_path + "labels.txt", "r") as f:
            classes = f.readlines()

        total_classes = len(classes)
        classes = sorted(classes)
        class_to_int = {}
        for idx, c in enumerate(classes):
            c = c.lower()
            c = c.replace(" ", "_")
            c = c.split("\n")[0]
            class_to_int[c] = idx

        with open(meta_path + "train.txt", "r") as f:
            training_files = f.readlines()

        with open(meta_path + "test.txt", "r") as f:
            test_files = f.readlines()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        training_labels = []
        for fil in training_files:
            training_labels.append(class_to_int[fil.split("/")[0]])

        test_labels = []
        for fil in test_files:
            test_labels.append(class_to_int[fil.split("/")[0]])
        # Make Dataloaders for train and test split

        training_files = [base_path + "images/" + fil.split("\n")[0] + ".jpg" for fil in training_files]
        test_files = [base_path + "images/" + fil.split("\n")[0] + ".jpg" for fil in test_files]
        train_dataset = CustomFoodDataSet(transform, training_files, training_labels)
        test_dataset = CustomFoodDataSet(transform, test_files, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        return train_dataloader, test_dataloader, test_dataloader, total_classes

    @staticmethod
    def prepare_caltech_dataset(base_path, test_size=0.2):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        folders = os.listdir(base_path)
        classes = [int(folder.split(".")[0]) for folder in folders]

        classes_to_files = {}
        for folder in folders:
            classes_to_files[folder] = os.listdir(base_path + folder)

        training_files = []
        training_labels = []
        test_files = []
        test_labels = []
        total_classes = len(classes)
        for folder in classes_to_files.keys():
            files = classes_to_files[folder]
            np.random.shuffle(files)
            seperator = int(len(files)*test_size)
            test_f = files[:seperator]
            train_f = files[seperator:]

            train_f = [base_path + folder + "/" + fil for fil in train_f]
            test_f = [base_path + folder + "/" + fil for fil in test_f]
            training_files.extend(train_f)
            training_labels.extend([int(folder.split(".")[0])-1] * len(train_f))
            test_files.extend(test_f)
            test_labels.extend([int(folder.split(".")[0])-1] * len(test_f))

        train_dataset = CustomCaltechDataSet(transform, training_files, training_labels)
        test_dataset = CustomCaltechDataSet(transform, test_files, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

        return train_dataloader, test_dataloader, test_dataloader, total_classes
