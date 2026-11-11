from sklearn.model_selection import train_test_split

images = []

for i in range(200):
    images.append("image" + str(i) + ".jpg")

labels = []

for i in range(100):
    labels.append("real")

for i in range(100):
    labels.append("fake")

train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.30, random_state=42, stratify=labels
)

validation_images, test_images, validation_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
)

def test_dataset_split_sizes():
    assert len(train_images) == 140
    assert len(validation_images) == 30
    assert len(test_images) == 30 