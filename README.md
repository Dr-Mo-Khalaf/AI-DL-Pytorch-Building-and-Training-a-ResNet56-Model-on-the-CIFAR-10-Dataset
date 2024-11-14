# AI-DL-Pytorch-Building-and-Training-a-ResNet56-Model-on-the-CIFAR-10-Dataset
Building and Training a ResNet-56 Model on the CIFAR-10 Dataset Using PyTorch for Deep Learning Applications step by step 

* report and summary done By Dr. Moh Khalaf
* All codes & implenmentation steps done by : Dr. Moh Khalaf

AI/DL Pytorch
Building-and-Training-a-ResNet56-Model-on-the-CIFAR-10-Dataset

**Project Overview for Weekly Team Meeting: ResNet-56 on CIFAR-10**

As a team leader, one of my responsibilities is to concisely explain the
fundamental concepts of the assigned project, which is ResNet-56, during
our weekly meetings. Here are the essential points my team should
understand:

**Report on AI-DL-PyTorch: Building and Training a ResNet56 Model on the
CIFAR-10 Dataset**

**Introduction**

The CIFAR-10 dataset, composed of 60,000 32x32 color images in 10
classes, is widely used in computer vision for image classification
tasks. It includes 50,000 images for training and 10,000 for testing.
For this project, a ResNet56 deep learning model was trained using
PyTorch to classify images in CIFAR-10 accurately. ResNet56, part of the
Residual Networks (ResNet) family, is a convolutional neural network
architecture that uses residual connections, helping to address the
degradation problem in deep networks, which occurs when adding layers
leads to higher training errors. The model was constructed using the
AI-DL-PyTorch framework, facilitating its implementation and training.

**Objective**

The primary objective of this project was to build and train a ResNet56
model to achieve high classification accuracy on the CIFAR-10 dataset.
This report covers the steps taken to build, train, and evaluate the
model, including an exploration of ResNet architecture, data
preprocessing, model construction, training configuration, and
evaluation results.

**1. Model Architecture - ResNet56**

### ResNet-56 with 3 Layers per Block
### ResNet-56
*  1. Residual Layers: 56 - 2 = 54 
*  2. Number of layers per stage: 54 / 3 = 18 
*  3. Number of blocks: 18 / 3 = 6
      
3 stages * 6 blocks/stage * 3 layers/block = 54 layers

ResNet56 is a version of the Residual Network (ResNet) architecture
specifically designed for deep learning tasks that benefit from
increased network depth. The ResNet architecture is based on the concept
of residual blocks, where the output from an earlier layer is added
directly to the output of a later layer, allowing information to bypass
intermediate layers. This helps prevent the \"vanishing gradient\"
problem by preserving gradient strength during backpropagation.

The ResNet56 architecture has 56 layers, with multiple residual blocks
throughout the network. Each block contains two or three convolutional
layers with batch normalization and ReLU activation functions. The
ResNet architecture\'s skip connections play a key role in enabling the
network to learn more complex patterns without performance degradation
due to increased depth.

**2. Data Preprocessing**

The CIFAR-10 dataset requires preprocessing to enhance model
performance. The preprocessing steps included:

-   **Normalization**: Images were normalized to a zero mean and unit
    variance to accelerate convergence during training.

-   **Data Augmentation**: Random transformations, including horizontal
    flipping and random cropping, were applied to increase the diversity
    of training images, thus reducing overfitting and improving
    generalization.

**3. Model Training**

The ResNet56 model was built and trained on the CIFAR-10 dataset using
the PyTorch library. Key steps in training included:

-   **Configuration Settings**: Hyperparameters such as learning rate,
    batch size, number of epochs, and optimizer were set based on
    experimentation and best practices for ResNet architectures. A
    learning rate of 0.02, batch size of 32, and training over **40
    epochs** were found effective.

-   **Optimizer and Loss Function**: The Stochastic Gradient Descent
    (SGD) optimizer with momentum was selected, which helps the model
    converge faster. Cross-entropy loss, a standard for multi-class
    classification, was used as the loss function.

-   **Learning Rate Scheduler**: A learning rate scheduler was
    implemented to decrease the learning rate by 50% every 10 epochs.
    This approach gradually reduces the step size in parameter updates
    as training progresses, allowing finer adjustments to weights, which
    enhances performance and promotes convergence.

**4. Training and Testing Process**

The model was trained on a GPU T4 system, which facilitated faster
processing and reduced training time. Key system specifications
included:

-   **System RAM**: 12.7 GB, of which 1.2 GB was utilized during
    training.

-   **GPU RAM**: 15.0 GB, with minimal usage of the available GPU
    memory, indicating that the system was well-suited for this model
    and dataset size.

During training, the model achieved **99% training accuracy** on the
GPU. However, testing was conducted on a CPU environment with an Intel
Core i7 processor and 8 GB of RAM, which likely contributed to a slower
inference time and potential discrepancies in performance. On the CPU,
the model achieved an **85% testing accuracy**.

This variation in accuracy between training and testing suggests a few
potential issues:

-   **Overfitting**: High training accuracy with relatively lower
    testing accuracy could indicate that the model learned specific
    patterns in the training set but failed to generalize as well on
    unseen data.

-   **Hardware Differences**: Testing on a CPU environment, especially
    with a smaller memory capacity, may result in slightly lower
    accuracy and increased inference time compared to a GPU setting.
    This can also lead to minor inconsistencies in performance if batch
    normalization statistics differ slightly.

**5. Evaluation and Results**

The ResNet56 model\'s performance was evaluated based on classification
accuracy and loss metrics. The **training accuracy reached 99% on the
GPU**, showcasing excellent learning capabilities, while the **testing
accuracy was 85% on the CPU**. Additional evaluation metrics like
confusion matrix, precision, recall, and F1-score were analyzed per
class to identify misclassifications and areas for potential
improvements.

**6. Conclusion**

Building and training a ResNet56 model on CIFAR-10 using AI-DL-PyTorch
demonstrated the effectiveness of deep residual learning for image
classification tasks. The model achieved high accuracy on the training
set, benefiting from the ResNet architecture's skip connections and
batch normalization techniques. Although testing accuracy was lower on
the CPU environment, it still shows competitive performance for
CIFAR-10.

The 99% training accuracy suggests the model learned the training data
effectively, but steps could be taken to improve generalization, such as
more aggressive regularization or further data augmentation. Training
the model over **40 epochs** proved to be a balance between adequate
learning and efficiency, though further tuning of epoch count may help
achieve even better generalization.

**Key Takeaways**

-   **ResNet56** effectively reduces the vanishing gradient problem in
    deep networks through residual blocks.

-   **Data Augmentation** and **Normalization** were crucial in
    enhancing model robustness and accuracy.

-   **Training Strategies** like mini-batch training with a batch size
    of 32, a scheduled learning rate reduction of 50% every 10 epochs,
    and limiting training to **40 epochs** significantly contributed to
    optimized performance.

-   **System Differences** between GPU and CPU environments can impact
    performance; testing accuracy on a CPU was 85%, whereas training on
    a GPU yielded 99% accuracy.

Overall, this project successfully demonstrated the capabilities of
ResNet56 on CIFAR-10, showcasing the potential of residual learning in
computer vision tasks, with opportunities for further improvement in
model generalization and adaptation across different hardware
environments.

**1. Why ResNet-56?**

-   **Deep Architecture**: ResNet-56 is a deep network with 56 layers,
    designed for accurate image classification.

-   **Residual Connections**: These \"skip connections\" address the
    vanishing gradient issue, allowing the network to go deeper without
    performance degradation.

**2. Dataset Basics - CIFAR-10**

-   **Images**: CIFAR-10 consists of 60,000 small (32x32 pixel) images,
    split across 10 classes.

-   **Training/Test Split**: 50,000 images are for training, and 10,000
    for testing, giving a comprehensive set for the model to learn and
    generalize.

**3. Data Preparation Steps**

-   **Normalization**: This scales pixel values to a standardized range,
    speeding up training.

-   **Augmentation**: Techniques like random flips and cropping enhance
    the model's ability to generalize by introducing variations.

**4. Training Configuration**

-   **Learning Rate**: Initially set at 0.02, decreasing by 50% every 10
    epochs to fine-tune updates.

-   **Mini-Batch Size**: Set at 32 for efficient memory use and smooth
    training.

-   **Epochs**: Training runs for 40 epochs, balancing learning depth
    with training time.

**5. Training and Testing Results**

-   **Training on GPU**: Achieved 99% accuracy on a T4 GPU, showing
    strong learning on the training data.

-   **Testing on CPU**: The model reached 85% accuracy on a CPU, which
    might indicate overfitting or differences due to hardware.

**6. Core Observations**

-   **Overfitting**: The gap between training and testing accuracy
    suggests overfitting, which can be reduced by increasing
    regularization or data augmentation.

-   **Hardware Impact**: Different hardware (GPU vs. CPU) can affect
    testing results slightly, so results on each may vary.

**7. Key Takeaways for Implementation**

-   **Learning Rate Scheduling**: Adjusting the learning rate over time
    supports optimal convergence.

-   **Residual Blocks**: These blocks in ResNet-56 are essential,
    allowing the network to train effectively by maintaining gradient
    strength across layers.

This overview provides the necessary foundation for understanding the
ResNet-56 project on CIFAR-10 and will support our discussions on model
performance and improvement areas in our weekly meetings.

