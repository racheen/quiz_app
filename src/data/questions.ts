import { Question } from '../types/question';

export const questions: Question[] = [
  {
    id: 1,
    topic: 'AML',
    question: 'The hyperplane of the SVM of a dataset with 3 features is a',
    options: ['point', 'line', 'plane', 'circle'],
    answer: 'plane',
    explanation:
      'The hyperplane in an SVM separates data points in an **N-dimensional space**. For **1 feature**, the hyperplane is a **point**. For **2 features**, the hyperplane is a **line**. For **3 features**, the hyperplane is a **plane** (a 2D surface in 3D space). Since the dataset has **3 features**, the SVM hyperplane is a **2D plane in 3D space**.',
  },
  {
    id: 2,
    topic: 'AML',
    question:
      'In an MLP, what it means when we specify hidden_layer_sizes = (5,3,2)',
    options: [
      '5 inputs, 2 outputs, 3 nodes in a layer between the inputs and outputs',
      '5 inputs, 2 outputs, and 3 layers between input and output layer',
      '3 hidden layers with 5 nodes in first layer, 3 nodes in second layer and 2 nodes in the third layer',
      '3 layers where 5 nodes in the input layer, 3 nodes in the hidden layer and 2 nodes in the output layer',
    ],
    answer:
      '3 hidden layers with 5 nodes in first layer, 3 nodes in second layer and 2 nodes in the third layer',
    explanation:
      'In an **MLP (Multi-Layer Perceptron)**, the `hidden_layer_sizes` parameter specifies the structure of the **hidden layers only** (not input or output layers). `hidden_layer_sizes = (5, 3, 2)` means: the **first hidden layer** has **5 neurons**, the **second hidden layer** has **3 neurons**, and the **third hidden layer** has **2 neurons**.',
  },
  {
    id: 3,
    topic: 'AML',
    question: 'What is the main role of the convolution operation?',
    options: [
      'create sequence of data from the given data',
      'reduce the image without losing the critical features',
      'technique to normalize the dataset',
      'technique to standardize the dataset',
    ],
    answer: 'reduce the image without losing the critical features',
    explanation:
      'In **Convolutional Neural Networks (CNNs)**, the **convolution operation** is used to: Extract **important features** from an image while preserving spatial relationships. Reduce the **dimensionality** of the image while retaining key information. Detect **edges, textures, patterns**, and other structures in an image.',
  },
  {
    id: 4,
    topic: 'AML',
    question:
      'The probability when a fair coin is tossed 2 times then at least one of them are heads:',
    options: ['1/4', '1/2', '3/4', '1'],
    answer: '3/4',
    explanation:
      'When tossing a fair coin **twice**, the possible outcomes are: HH, HT, TH, TT. The event **"at least one head"** means we consider all cases except **TT**. Only **TT** does not satisfy the condition (1 outcome). The total number of outcomes = **4**. The favorable outcomes = **HH, HT, TH** = **3**.',
  },
  {
    id: 5,
    topic: 'AML',
    question: 'Image captioning is which type of RNN?',
    options: ['Many to One', 'One to One', 'Many to Many', 'One to Many'],
    answer: 'One to Many',
    explanation:
      'In **image captioning**, an image is input into a **CNN** to extract features. These features are then passed into an **RNN (LSTM/GRU)**, which **generates a multiple textual description** (caption). Since the input is an **image (a set of features)** and the output is **one sequence (caption)**, it follows a **One-to-Many** architecture.',
  },
  {
    id: 6,
    topic: 'AML',
    question: 'PCA is',
    options: [
      'unsupervised learning algorithm',
      'semi-supervised learning algorithm',
      'reinforcement learning algorithm',
      'supervised learning algorithm',
    ],
    answer: 'unsupervised learning algorithm',
    explanation:
      'Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It does not require labels, meaning it does not rely on supervision, making it an unsupervised learning algorithm.',
  },
  {
    id: 7,
    topic: 'AML',
    question: 'The dimension of the hyperplane of an SVM is based on',
    options: [
      'number of features',
      'number of classes',
      'number of instances',
      'All of these',
    ],
    answer: 'number of features',
    explanation:
      'In **Support Vector Machines (SVM)**, the **hyperplane** is a decision boundary that separates different classes in the feature space. The dimension of the hyperplane depends on the number of **features** in the dataset.',
  },
  {
    id: 8,
    topic: 'AML',
    question: 'The hyperplane of SVM is generated based on',
    options: [
      'all instances',
      'n instances, where n is the number of features',
      'instances that are closer to the inner boundary (hyperplane)',
      '10% of the instances',
    ],
    answer: 'instances that are closer to the inner boundary (hyperplane)',
    explanation:
      'In **Support Vector Machines (SVM)**, the hyperplane is determined by **support vectors**, which are the instances closest to the decision boundary. These **support vectors** play a crucial role in defining the optimal separating hyperplane.',
  },
  {
    id: 9,
    topic: 'AML',
    question: 'The range of the output of a ReLU activation function?',
    options: ['[-1, 1]', '[0, 1]', '[0, ∞ ]', '[-∞ , ∞ ]'],
    answer: '[0, ∞ ]',
    explanation:
      'The **Rectified Linear Unit (ReLU)** activation function is defined as: $$ f(x) =max(0,x) $$ This means: If x<0, the output is **0**. If x≥0, the output is **x**. Thus, the output **ranges from 0 to infinity**: **[0, ∞]**.',
  },
  {
    id: 10,
    topic: 'AML',
    question: 'What is the advantage of using CNN?',
    options: [
      'PCA works really well',
      'No need to do manual preprocessing',
      'Gives highest accuracy among all algorithms for all supervised learning',
      'LDA works really well',
    ],
    answer: 'No need to do manual preprocessing',
    explanation:
      'Convolutional Neural Networks (CNNs) are particularly powerful for tasks involving image or spatial data. One of their key advantages is their ability to automatically learn and extract hierarchical features from raw data (such as images), which eliminates the need for extensive manual feature extraction or preprocessing.',
  },
  {
    id: 11,
    topic: 'AML',
    question: 'Logistic regression is for',
    options: [
      'Outlier Detection',
      'Clustering',
      'regression',
      'classification',
    ],
    answer: 'classification',
    explanation:
      "Logistic Regression is a classification algorithm. It is used to model the probability of a binary outcome (1 or 0) and makes predictions based on a logistic (sigmoid) function. It's widely used for binary classification tasks.",
  },
  {
    id: 12,
    topic: 'AML',
    question: 'The increase in the number of patients due to frostbite is',
    options: ['seasonal variation', 'trend', 'irregular variation', 'residue'],
    answer: 'seasonal variation',
    explanation:
      'The increase in the number of patients due to frostbite is an example of seasonal variation. Seasonal variations are regular and predictable changes that occur at certain times of the year.',
  },
  {
    id: 13,
    topic: 'AML',
    question:
      'We have an image of size 7x7 with valid padding, on which a filter of size 3x3 is applied with a stride of 2. What will be the size of the convoluted matrix?',
    options: ['4x4', '6x6', '5x5', '3x3'],
    answer: '3x3',
    explanation:
      'For **valid padding**, the output size is calculated using the formula: $$ \text{Output Size} = left( \frac{\text{Input Size} - \text{Filter Size}}{\text{Stride}} \right) + 1 $$ For this case: $$ \text{Output Size} = left( \frac{7 - 3}{2} \right) + 1 = 3 $$.',
  },
  {
    id: 14,
    topic: 'AML',
    question:
      'We have an input image of size 28x28. A filter of size 7x7 is applied with a stride of 1. What will be the size of the convoluted matrix?',
    options: ['21x21', '35x35', '22x22', '25x25'],
    answer: '22x22',
    explanation:
      'With a **filter size of 7x7** and **stride 1**, the output size is calculated using the same formula: $$ \text{Output Size} = left( \frac{\text{Input Size} - \text{Filter Size}}{\text{Stride}} \right) + 1 $$ $$ \text{Output Size} = left( \frac{28 - 7}{1} \right) + 1 = 22 $$.',
  },
];
