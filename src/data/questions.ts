import { Question } from '../types/question';

export enum TopicEnum {
  AML = 'Advanced Machine Leraning',
  DataPreprocessing = 'Data Preprocessing',
  SupportVectorMachines = 'Support Vector Machines',
  NeuralNetworks = 'Neural Networks',
  TimeSeriesRNN = 'Time Series RNN',
  NaiveBayes = 'Naive Bayes',
  Clustering = 'Clustering',
  HyperparameterTuning = 'Hyperparameter Tuning',
  Visualizations = 'Visualizations',
  ClassifierFusion = 'Classifier Fusion',
  ScikitLearn = 'Scikit Learn',
  MV = 'Machine Vision',
  PyTorch = 'PyTorch',
  ObjectDetection = 'Object Detection',
  ObjectTracking = 'Object Tracking',
  SensorFusion = 'Sensor Fusion',
  MVIntro = 'Machine Vision Introduction',
  ImageProcessing = 'Fundamentals of Image Processing',
  SegmentationOD = 'Segmentation and Object Detection',
  CNN = 'CNN in Machine Vision',
  DLCNN = 'Deep Learning for Image Classification',
  RL = 'Reinforcement Learning',
}
export const MainTopic = {
  MV: TopicEnum.MV,
  AML: TopicEnum.AML,
  RL: TopicEnum.RL,
} as const;

export type MainTopic = typeof MainTopic[keyof typeof MainTopic];

export const questions: Question[] = [
  {
    id: 1,
    topic: [TopicEnum.AML, TopicEnum.SupportVectorMachines],
    question: 'The hyperplane of the SVM of a dataset with 3 features is a',
    options: ['point', 'line', 'plane', 'circle'],
    answer: 'plane',
    explanation:
      'The hyperplane in an SVM separates data points in an **N-dimensional space**. For **1 feature**, the hyperplane is a **point**. For **2 features**, the hyperplane is a **line**. For **3 features**, the hyperplane is a **plane** (a 2D surface in 3D space). Since the dataset has **3 features**, the SVM hyperplane is a **2D plane in 3D space**.',
  },
  {
    id: 2,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
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
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
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
    topic: [TopicEnum.AML, TopicEnum.NaiveBayes],
    question:
      'The probability when a fair coin is tossed 2 times then at least one of them are heads:',
    options: ['1/4', '1/2', '3/4', '1'],
    answer: '3/4',
    explanation:
      'When tossing a fair coin **twice**, the possible outcomes are: HH, HT, TH, TT. The event **"at least one head"** means we consider all cases except **TT**. Only **TT** does not satisfy the condition (1 outcome). The total number of outcomes = **4**. The favorable outcomes = **HH, HT, TH** = **3**.',
  },
  {
    id: 5,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question: 'Image captioning is which type of RNN?',
    options: ['Many to One', 'One to One', 'Many to Many', 'One to Many'],
    answer: 'One to Many',
    explanation:
      'In **image captioning**, an image is input into a **CNN** to extract features. These features are then passed into an **RNN (LSTM/GRU)**, which **generates a multiple textual description** (caption). Since the input is an **image (a set of features)** and the output is **one sequence (caption)**, it follows a **One-to-Many** architecture.',
  },
  {
    id: 6,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'PCA is',
    options: [
      'unsupervised learning algorithm',
      'semi-supervised learning algorithm',
      'reinforcement learning algorithm',
      'supervised learning algorithm',
    ],
    answer: 'unsupervised learning algorithm',
    explanation:
      'Principal Component Analysis (PCis a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It does not require labels, meaning it does not rely on supervision, making it an unsupervised learning algorithm.',
  },
  {
    id: 7,
    topic: [TopicEnum.AML, TopicEnum.SupportVectorMachines],
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
    topic: [TopicEnum.AML, TopicEnum.SupportVectorMachines],
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
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question: 'The range of the output of a ReLU activation function?',
    options: ['[-1, 1]', '[0, 1]', '[0, ∞ ]', '[-∞ , ∞ ]'],
    answer: '[0, ∞ ]',
    explanation:
      'The **Rectified Linear Unit (ReLU)** activation function is defined as: $ f(x) =max(0,x) $ This means: If x<0, the output is **0**. If x≥0, the output is **x**. Thus, the output **ranges from 0 to infinity**: **[0, ∞]**.',
  },
  {
    id: 10,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
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
    topic: [TopicEnum.AML],
    question: 'Logistic regression is for',
    options: [
      'Outlier Detection',
      'Clustering',
      'regression',
      'classification',
    ],
    answer: 'classification',
    explanation:
      "Logistic Regression is a classification algorithm. It is used to model the probability of a binary outcome (1 or 0) and makes predictions based on a logistic (sigmoifunction. It's widely used for binary classification tasks.",
  },
  {
    id: 12,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question: 'The increase in the number of patients due to frostbite is',
    options: ['seasonal variation', 'trend', 'irregular variation', 'residue'],
    answer: 'seasonal variation',
    explanation:
      'The increase in the number of patients due to frostbite is an example of seasonal variation. Seasonal variations are regular and predictable changes that occur at certain times of the year.',
  },
  {
    id: 13,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question:
      'We have an image of size 7x7 with valid padding, on which a filter of size 3x3 is applied with a stride of 2. What will be the size of the convoluted matrix?',
    options: ['4x4', '6x6', '5x5', '3x3'],
    answer: '3x3',
    explanation:
      'For **valid padding**, the output size is calculated using the formula: $\\text{Output Size} = \\left( \\frac{\\text{Input Size} - \text{Filter Size}}{\text{Stride}} \\right) + 1$ For this case: $\\text{Output Size} = \\left(\\frac{7 - 3}{2}\\right) + 1 = 3$.',
  },
  {
    id: 14,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question:
      'We have an input image of size 28x28. A filter of size 7x7 is applied with a stride of 1. What will be the size of the convoluted matrix?',
    options: ['21x21', '35x35', '22x22', '25x25'],
    answer: '22x22',
    explanation:
      'With a **filter size of 7x7** and **stride 1**, the output size is calculated using the same formula: $\\text{Output Size} = \\left( \\frac{\\text{Input Size} - \\text{Filter Size}}{\\text{Stride}} \\right) + 1$ $\\text{Output Size} = \\left( \\frac{28 - 7}{1} \\right) + 1 = 22$.',
  },
  {
    id: 15,
    topic: [TopicEnum.AML, TopicEnum.SupportVectorMachines],
    question: 'Support Vectors can influence',
    options: [
      'Position of the hyper plane',
      'Position and orientation of the hyperplane',
      'Orientation of the hyperplane',
      'thickness of the hyperplane',
    ],
    answer: 'Position and orientation of the hyperplane',
    explanation:
      'Support Vectors are the data points that are closest to the decision boundary (or hyperplane) in Support Vector Machines (SVM). They play a crucial role in defining the position and orientation of the hyperplane because the hyperplane is determined in such a way that it maximizes the margin (distance) between the support vectors of the two classes. Position and orientation of the hyperplane: The support vectors influence where the hyperplane is placed (its position) and the angle at which it is oriented (its orientation). By adjusting the position of the support vectors, the hyperplane can shift or rotate to better separate the data points.',
  },
  {
    id: 16,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question: 'Same padding means',
    options: [
      'add 1 padding on both sides',
      'No padding',
      'add padding such that the convoluted output matrix size should be the same as the input matrix size',
      'if the image size is nxn, add same n pixels on both sides',
    ],
    answer:
      'add padding such that the convoluted output matrix size should be the same as the input matrix size',
    explanation:
      'Same padding means that padding is added to the input image in such a way that the output size of the convolution is the same as the input size. The padding ensures that the filter can slide across the entire image without reducing the spatial dimensions of the input.',
  },
  {
    id: 17,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question:
      'The number of nodes in the input layer is 10 and the hidden layer is 6. The maximum number of connections from the input layer to the hidden layer is',
    options: [
      'it depends on how we design the network',
      'more than 60',
      'less than 60',
      '60',
    ],
    answer: '60',
    explanation:
      'In a feedforward neural network, the number of connections between layers is calculated by multiplying the number of nodes in the input layer by the number of nodes in the hidden layer. Here, the number of nodes in the input layer is 10 and in the hidden layer is 6. Therefore, the maximum number of connections from the input layer to the hidden layer is:\n\n$$\\text{Number of connections} = \\text{Input nodes x Hidden nodes} = 10 \\text{ x } 6 = 60$$',
  },
  {
    id: 18,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'If your dataset is unlabeled, which technique is the best to reduce the dimensionality of this dataset?',
    options: [
      'LDA',
      'PCA',
      'Cannot tell, it depends on the attributes',
      'both are equivalent',
    ],
    answer: 'PCA',
    explanation:
      'PCA (Principal Component Analysis) is an unsupervised technique used to reduce the dimensionality of data. It does not require labeled data and works by identifying the directions (principal components) in which the data varies the most, and projecting the data onto these directions to reduce its dimensionality.',
  },

  {
    id: 19,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question:
      'For a neural network, which of the following affects the trade-off between underfitting and overfitting?',
    options: [
      'number of hidden nodes',
      'initial choice of weights',
      'bias',
      'learning rate',
    ],
    answer: 'number of hidden nodes',
    explanation:
      'The number of hidden nodes in a neural network directly affects the model’s complexity. Too few nodes can cause underfitting, while too many can cause overfitting. Thus, the number of hidden nodes helps strike a balance between underfitting and overfitting.',
  },
  {
    id: 20,
    topic: [TopicEnum.AML, TopicEnum.NaiveBayes],
    question: 'What are the strong assumptions that we make in Naïve Bayes:',
    options: [
      'Features are dependent, numerical data to be converted to categorical',
      'Features are independent, numerical data follows gaussian distribution',
      'Features are dependent, numerical data follows normal distribution',
      'Features are independent, numerical data to be converted to categorical',
    ],
    answer:
      'Features are independent, numerical data follows gaussian distribution',
    explanation:
      'Naïve Bayes assumes that all features are conditionally independent given the class label. For numerical features, it assumes they follow a Gaussian (normal) distribution within each class. These simplifying assumptions make the algorithm computationally efficient.',
  },
  {
    id: 21,
    topic: [TopicEnum.AML, TopicEnum.SupportVectorMachines],
    question: 'The hyperplane of the SVM of a dataset with 2 features is a',
    options: ['line', 'point', 'circle', 'plane'],
    answer: 'line',
    explanation:
      'In a dataset with 2 features, the data lies in a 2D plane. In this case, the SVM hyperplane is a line that separates the classes.',
  },
  {
    id: 22,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'If noise dominates in your dataset, which technique is the best to reduce the dimensionality of this dataset?',
    options: [
      'both are equivalent',
      'LDA',
      'Cannot tell, it depends on the attributes',
      'PCA',
    ],
    answer: 'LDA',
    explanation:
      'LDA (Linear Discriminant Analysis) is a supervised dimensionality reduction technique that maximizes class separability. When labels are available and the dataset contains noise, LDA often performs better than PCA because it focuses on preserving class-discriminative information rather than just maximizing variance like PCA.',
  },
  {
    id: 23,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question: 'CNN is best suited for',
    options: [
      'Temperature dataset with daily temperatures from Jan-Dec 2022',
      'Cars dataset with 64x64 images of 50000 cars',
      'dataset with 10000 reviews to predict the sentiment',
      'Titanic dataset with details of 900 travelers',
    ],
    answer: 'Cars dataset with 64x64 images of 50000 cars',
    explanation:
      'CNNs (Convolutional Neural Networks) are best suited for image data because they can learn spatial hierarchies of features, making them highly effective at processing visual data.',
  },
  {
    id: 24,
    topic: [TopicEnum.AML, TopicEnum.NeuralNetworks],
    question: 'What is the main role of Pooling?',
    options: [
      'extract dominant features',
      'technique to standardize the dataset',
      'technique to normalize the dataset',
      'create sequence of data from the given data',
    ],
    answer: 'extract dominant features',
    explanation:
      'Pooling is a downsampling operation in CNNs that reduces the spatial dimensions of feature maps while retaining the most important information. It helps extract dominant features and reduces computational load.',
  },
  {
    id: 25,
    topic: [TopicEnum.AML, TopicEnum.NaiveBayes],
    question: 'The conditional probability P(A|B) is',
    options: [
      '$P(A|B)= \\frac{P(A∪B)}{P(B)}$',
      '$P(A|B)= \\frac{P(A∩B)}{P(B)}$',
      '$P(A|B)= \\frac{P(A∩B)}{P(A)}$',
      '$P(A|B)= \\frac{P(A∪B)}{P(A)}$',
    ],
    answer: '$P(A|B)= \\frac{P(A∩B)}{P(B)}$',
    explanation:
      'Conditional probability is defined as the probability of event A given that event B has occurred. It is calculated using the formula: \n\n$P(A|B)= \\frac{P(A∩B)}{P(B)}$',
  },
  {
    id: 26,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'What is the primary goal of the project, and how will the insights from the data help address the business problem?',
    options: [
      'To explore data patterns without a clear business goal',
      'To identify actionable insights that can drive business decisions',
      'To collect large amounts of data for storage',
      'To improve internal processes without involving data',
    ],
    answer: 'To identify actionable insights that can drive business decisions',
    explanation:
      'The primary goal is to use data analysis to derive insights that can influence business strategy and improve outcomes.',
  },
  {
    id: 27,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'What are the key metrics or performance indicators that will define the success of the project?',
    options: [
      'Data storage size and processing speed',
      'Business revenue increase and customer satisfaction',
      'Accuracy of data labeling',
      'Complexity of the model used',
    ],
    answer: 'Business revenue increase and customer satisfaction',
    explanation:
      'Key performance indicators should focus on measurable outcomes that directly impact business success, such as revenue and customer satisfaction.',
  },
  {
    id: 28,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'What types of data sources are available, and how reliable are they for the intended analysis?',
    options: [
      'Only internal company data, which is highly reliable',
      'Multiple data sources, including both internal and external, with varied reliability',
      'Only publicly available data, which is mostly unreliable',
      'Data sources are not important as long as the analysis is conducted',
    ],
    answer:
      'Multiple data sources, including both internal and external, with varied reliability',
    explanation:
      'Using both internal and external data allows for a comprehensive analysis, but the reliability of these sources must be assessed.',
  },
  {
    id: 29,
    topic: [TopicEnum.AML],
    question:
      'How do you plan to identify and address any missing or incomplete data in the dataset?',
    options: [
      'Ignore missing data and proceed with analysis',
      'Remove any records with missing data',
      'Impute missing data using mean, median, or mode for numerical and categorical columns',
      'Only focus on the complete data without any imputation',
    ],
    answer:
      'Impute missing data using mean, median, or mode for numerical and categorical columns',
    explanation:
      'Imputation is a common approach for dealing with missing data, replacing missing values with statistical measures like mean, median, or mode.',
  },
  {
    id: 30,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'What visualizations and statistical analyses will be conducted to uncover trends and patterns in the data?',
    options: [
      'Scatter plots and pie charts for basic insights',
      'Heatmaps, histograms, and boxplots for deeper understanding',
      'No visualizations, as the focus is on raw data',
      'Only bar graphs to compare categories',
    ],
    answer: 'Heatmaps, histograms, and boxplots for deeper understanding',
    explanation:
      'These visualizations allow for a better understanding of distributions, relationships, and outliers in the data.',
  },
  {
    id: 31,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'How will you test for correlations between features, and what steps will be taken if strong associations are found?',
    options: [
      'Use scatter plots and Pearson correlation coefficient, then drop correlated features',
      "Ignore correlations, as they don't impact the model's performance",
      'Conduct hypothesis testing without addressing correlations',
      'Look only for correlations between numeric features, ignoring categorical ones',
    ],
    answer:
      'Use scatter plots and Pearson correlation coefficient, then drop correlated features',
    explanation:
      "Testing correlations using Pearson's coefficient helps identify multicollinearity, which may lead to dropping highly correlated features.",
  },
  {
    id: 32,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'What criteria will you use to select the most relevant features for the model?',
    options: [
      'Randomly select features',
      'Use correlation and association analysis, followed by domain knowledge',
      'Choose features based only on their data type',
      'Use all available features without filtering',
    ],
    answer:
      'Use correlation and association analysis, followed by domain knowledge',
    explanation:
      'Feature selection should be based on statistical methods and domain expertise to improve model performance and interpretability.',
  },
  {
    id: 33,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'How will you handle noise or errors in the dataset during the data cleaning process?',
    options: [
      'Remove all records with noise or errors',
      'Use data imputation to replace erroneous values',
      "Ignore noise and errors, as they don't significantly impact results",
      'Adjust the dataset by removing duplicate records and outliers',
    ],
    answer: 'Adjust the dataset by removing duplicate records and outliers',
    explanation:
      'Removing duplicates and outliers helps improve the quality of the data, ensuring that the analysis is not skewed by erroneous entries.',
  },
  {
    id: 34,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'What machine learning algorithms (supervised or unsuperviseare being considered, and why are they appropriate for the project?',
    options: [
      'Only unsupervised learning methods, as there are no labeled data',
      'Both supervised and unsupervised algorithms, chosen based on data structure and goal',
      'Only regression models, as the goal is predicting continuous values',
      'Only decision trees, as they are the most straightforward',
    ],
    answer:
      'Both supervised and unsupervised algorithms, chosen based on data structure and goal',
    explanation:
      'The choice of algorithms depends on the problem type (classification, regression, clustering) and the data available (labeled or unlabeled).',
  },
  {
    id: 35,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      "How will you evaluate the model's performance, and what metrics will be used to assess accuracy, precision, and recall?",
    options: [
      'Evaluate performance based solely on accuracy',
      'Use a combination of cross-validation, confusion matrix, and performance metrics like precision, recall, and F1 score',
      'Focus only on precision and ignore recall',
      'Use accuracy and ignore precision and recall for simplicity',
    ],
    answer:
      'Use a combination of cross-validation, confusion matrix, and performance metrics like precision, recall, and F1 score',
    explanation:
      'A comprehensive evaluation includes cross-validation and a combination of metrics like precision, recall, and F1 score to fully understand model performance.',
  },
  {
    id: 36,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: 'In Power BI, what is the purpose of the DAX language?',
    options: [
      'It is used for designing the visual interface of reports.',
      'It is used for importing and transforming data.',
      'It is used to create custom calculations and measures in reports.',
      'It is used for connecting Power BI to external databases.',
    ],
    answer: 'It is used to create custom calculations and measures in reports.',
    explanation:
      'DAX (Data Analysis Expressions) is a formula language used in Power BI for creating custom calculations, measures, and calculated columns in reports.',
  },
  {
    id: 37,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'What type of data sources can Power BI connect to for data analysis?',
    options: [
      'Only Excel files',
      'Only SQL databases',
      'A wide variety of data sources, including Excel, SQL, web services, and more',
      'Only cloud-based data sources',
    ],
    answer:
      'A wide variety of data sources, including Excel, SQL, web services, and more',
    explanation:
      'Power BI supports connections to a wide range of data sources, including Excel files, SQL databases, web services, cloud services, and more.',
  },
  {
    id: 38,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: 'How can you refresh data in Power BI?',
    options: [
      'Manually import the data again',
      'Use Power Query Editor to refresh the data',
      'Set up automatic refresh schedules on Power BI Service',
      'Data cannot be refreshed once imported',
    ],
    answer: 'Set up automatic refresh schedules on Power BI Service',
    explanation:
      'Power BI allows users to set up scheduled refreshes to automatically refresh the data on the Power BI Service, ensuring reports are up-to-date.',
  },
  {
    id: 39,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: "What is the purpose of Tableau's 'Calculated Field'?",
    options: [
      'To visualize data trends',
      'To filter the data',
      'To create new data from existing fields using formulas or expressions',
      'To import external data into Tableau',
    ],
    answer:
      'To create new data from existing fields using formulas or expressions',
    explanation:
      'A calculated field in Tableau allows you to create new data by applying formulas or expressions to existing data fields.',
  },
  {
    id: 40,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'Which type of join can Tableau use when combining data from multiple tables?',
    options: ['Left Join', 'Inner Join', 'Outer Join', 'All of the above'],
    answer: 'All of the above',
    explanation:
      'Tableau supports multiple types of joins when combining data from multiple tables, including left join, inner join, and outer join, depending on the analysis requirements.',
  },
  {
    id: 41,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'How can you ensure that Tableau visualizations are interactive for the user?',
    options: [
      'By using only static charts',
      'By adding filters, parameters, and dashboard actions',
      'By limiting the number of charts in a dashboard',
      'By exporting visualizations as static images',
    ],
    answer: 'By adding filters, parameters, and dashboard actions',
    explanation:
      'Interactive visualizations in Tableau are achieved by adding interactive elements such as filters, parameters, and dashboard actions that allow users to explore the data.',
  },
  {
    id: 42,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'Which feature of Tableau allows you to combine data from multiple sources into a single view?',
    options: [
      'Data Blending',
      'Data Extract',
      'Dashboard Actions',
      'Calculated Fields',
    ],
    answer: 'Data Blending',
    explanation:
      'Data Blending in Tableau allows you to combine data from different sources into a single view, enabling analysis from multiple data sets.',
  },
  {
    id: 43,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: "What is Power BI's 'Report'?",
    options: [
      'A snapshot of the data',
      'A collection of multiple visuals, tables, and charts based on a dataset',
      'A simple chart visual',
      'A process to clean and transform data',
    ],
    answer:
      'A collection of multiple visuals, tables, and charts based on a dataset',
    explanation:
      'A Power BI report is a collection of visualizations like charts, graphs, and tables, all based on data from a single or multiple datasets, used to represent insights.',
  },
  {
    id: 44,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: "What is Power BI's 'Power Query' used for?",
    options: [
      'It is used for data visualization',
      'It is used for creating custom reports',
      'It is used for importing, transforming, and cleaning data',
      'It is used for defining the relationships between datasets',
    ],
    answer: 'It is used for importing, transforming, and cleaning data',
    explanation:
      'Power Query in Power BI is used for importing, transforming, and cleaning data before it is loaded into the model for analysis.',
  },
  {
    id: 45,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: "What is Tableau's 'Extract' feature used for?",
    options: [
      'To remove unnecessary data from the dataset',
      'To create a snapshot of the data that improves performance',
      'To automatically update data on a schedule',
      'To combine data from multiple sources',
    ],
    answer: 'To create a snapshot of the data that improves performance',
    explanation:
      "Tableau's Extract feature creates a snapshot of the data, improving performance by reducing the need to query live data sources.",
  },
  {
    id: 46,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'What type of clustering algorithm does kMeans represent?',
    options: [
      'Partitional clustering',
      'Density-based clustering',
      'Distribution-based clustering',
      'Hierarchical clustering',
    ],
    answer: 'Partitional clustering',
    explanation:
      'kMeans is a partitional clustering algorithm that assigns data points to clusters based on their proximity to centroids.',
  },
  {
    id: 47,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question:
      'Which clustering algorithm groups data based on density of points in a region?',
    options: [
      'DBSCAN',
      'kMeans',
      'Agglomerative Clustering',
      'Expectation-Maximization',
    ],
    answer: 'DBSCAN',
    explanation:
      'DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups data based on the density of data points in a region.',
  },
  {
    id: 48,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question:
      'In DBSCAN, what do we call a point that has fewer than the minimum required neighbors within a given radius?',
    options: ['Noise point', 'Core point', 'Border point', 'Centroid'],
    answer: 'Noise point',
    explanation:
      'In DBSCAN, a noise point is one that does not meet the criteria to be a core point or border point.',
  },
  {
    id: 49,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question:
      'Which of the following clustering types assigns probabilities to data points belonging to each cluster?',
    options: [
      'Soft clustering',
      'Hard clustering',
      'Hierarchical clustering',
      'Partitional clustering',
    ],
    answer: 'Soft clustering',
    explanation:
      'Soft clustering, used in models like Expectation-Maximization, assigns each point a probability of belonging to each cluster.',
  },
  {
    id: 50,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question:
      'Which clustering method builds a tree-like structure without requiring a predefined number of clusters?',
    options: [
      'Hierarchical clustering',
      'Partitional clustering',
      'kMeans',
      'DBSCAN',
    ],
    answer: 'Hierarchical clustering',
    explanation:
      'Hierarchical clustering creates a dendrogram and does not require predefining the number of clusters.',
  },
  {
    id: 51,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'What does the Expectation step in the EM algorithm do?',
    options: [
      'Estimates the probability of each point belonging to a cluster',
      'Updates cluster centroids',
      'Removes noise points',
      'Determines the number of clusters',
    ],
    answer: 'Estimates the probability of each point belonging to a cluster',
    explanation:
      'The Expectation step calculates the probability that each data point belongs to each of the clusters.',
  },
  {
    id: 52,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question:
      'In hierarchical clustering, which linkage method considers the shortest distance between points in two clusters?',
    options: [
      'Single linkage',
      'Complete linkage',
      'Average linkage',
      'Centroid linkage',
    ],
    answer: 'Single linkage',
    explanation:
      'Single linkage determines the distance between two clusters based on the shortest distance between any two points in the clusters.',
  },
  {
    id: 53,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'Which statement best describes the limitation of kMeans?',
    options: [
      'It requires the number of clusters to be specified beforehand.',
      'It works well with clusters of arbitrary shapes.',
      'It performs soft clustering by default.',
      'It uses a probabilistic model to cluster data.',
    ],
    answer: 'It requires the number of clusters to be specified beforehand.',
    explanation:
      'kMeans needs the user to specify k, the number of clusters, which may not always be known.',
  },
  {
    id: 54,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'What are the core components used to classify points in DBSCAN?',
    options: [
      'Core, Border, Noise',
      'Centroid, Radius, Mean',
      'Gaussian, Mean, Variance',
      'Hard, Soft, Fuzzy',
    ],
    answer: 'Core, Border, Noise',
    explanation:
      'DBSCAN categorizes points as Core, Border, or Noise based on density and neighborhood radius.',
  },
  {
    id: 55,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      'What is the main purpose of hyperparameter tuning in machine learning?',
    options: [
      'To select the set of optimal hyperparameters that improves model accuracy, prevents overfitting/underfitting, and ensures efficient resource use',
      'To adjust model weights and biases during training',
      'To add more features to the dataset',
      "To optimize the model's computational resources",
    ],
    answer:
      'To select the set of optimal hyperparameters that improves model accuracy, prevents overfitting/underfitting, and ensures efficient resource use',
    explanation:
      "Hyperparameter tuning is crucial for improving a model's performance and ensuring it generalizes well to unseen data.",
  },
  {
    id: 56,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question: 'How do hyperparameters differ from model parameters?',
    options: [
      'Hyperparameters are set before training, while parameters are learned during training',
      'Hyperparameters are learned during training, while parameters are set before training',
      'Both hyperparameters and parameters are learned during training',
      'Both hyperparameters and parameters are set before training',
    ],
    answer:
      'Hyperparameters are set before training, while parameters are learned during training',
    explanation:
      'Hyperparameters define model configuration before training, whereas parameters (like weights) are learned as the model trains.',
  },
  {
    id: 57,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      "Why is tuning hyperparameters important for a machine learning model's performance?",
    options: [
      'It improves model accuracy and prevents overfitting/underfitting',
      'It reduces the number of features in the model',
      'It helps increase the size of the training dataset',
      'It changes the model’s underlying architecture',
    ],
    answer: 'It improves model accuracy and prevents overfitting/underfitting',
    explanation:
      'Proper hyperparameter tuning ensures optimal performance and reduces the risk of overfitting or underfitting.',
  },
  {
    id: 58,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      'Which hyperparameter tuning technique tests all possible combinations of specified hyperparameter values?',
    options: [
      'Grid Search',
      'Random Search',
      'Manual Search',
      'Bayesian Optimization',
    ],
    answer: 'Grid Search',
    explanation:
      'Grid Search exhaustively evaluates all possible combinations of specified hyperparameters to find the optimal set.',
  },
  {
    id: 59,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question: 'What does `GridSearchCV` do in scikit-learn?',
    options: [
      'It performs an exhaustive search over all specified hyperparameter values using cross-validation',
      'It randomly selects a few hyperparameters for evaluation',
      'It fine-tunes model parameters during training',
      'It splits the dataset into multiple subsets for parallel processing',
    ],
    answer:
      'It performs an exhaustive search over all specified hyperparameter values using cross-validation',
    explanation:
      '`GridSearchCV` is used to search over a range of hyperparameters and determine the best combination for the model.',
  },
  {
    id: 60,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      'In the context of `GridSearchCV`, what does `param_grid` contain?',
    options: [
      'A dictionary specifying the hyperparameters and their candidate values for tuning',
      'A list of the training data subsets',
      'The range of possible model outputs',
      'A function to calculate model loss',
    ],
    answer:
      'A dictionary specifying the hyperparameters and their candidate values for tuning',
    explanation:
      '`param_grid` is a dictionary that defines the hyperparameters to search over and their possible values during grid search.',
  },
  {
    id: 61,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      'What is the purpose of `best_params_` in both `GridSearchCV` and `RandomizedSearchCV`?',
    options: [
      'It stores the combination of hyperparameters that gave the best performance during cross-validation',
      'It computes the best training data subset for evaluation',
      "It validates the model's final performance after training",
      'It specifies the final model architecture after tuning',
    ],
    answer:
      'It stores the combination of hyperparameters that gave the best performance during cross-validation',
    explanation:
      '`best_params_` contains the optimal hyperparameter values that produced the best performance during cross-validation.',
  },
  {
    id: 62,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      'What is the difference between `GridSearchCV` and `RandomizedSearchCV`?',
    options: [
      '`GridSearchCV` evaluates all possible combinations, while `RandomizedSearchCV` samples a fixed number of random combinations',
      '`GridSearchCV` is faster than `RandomizedSearchCV`',
      '`RandomizedSearchCV` always uses more hyperparameter values than `GridSearchCV`',
      '`GridSearchCV` is only available for classification models',
    ],
    answer:
      '`GridSearchCV` evaluates all possible combinations, while `RandomizedSearchCV` samples a fixed number of random combinations',
    explanation:
      '`GridSearchCV` performs a comprehensive search, whereas `RandomizedSearchCV` randomly samples combinations, making it more efficient in large search spaces.',
  },
  {
    id: 63,
    topic: [TopicEnum.AML, TopicEnum.HyperparameterTuning],
    question:
      'What Python module is commonly used to define random distributions for `RandomizedSearchCV`?',
    options: ['scipy.stats', 'numpy', 'matplotlib', 'sklearn.linear_model'],
    answer: 'scipy.stats',
    explanation:
      '`scipy.stats` provides random distributions like `randint` that are used in `RandomizedSearchCV` for defining the search space.',
  },
  {
    id: 64,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'What is the main effect of high bias in a machine learning model?',
    options: [
      'The model underfits and fails to capture important patterns',
      'The model overfits and captures noise along with patterns',
      'The model performs excellently on both training and test data',
      'The model has high variance and fluctuates with small changes in data',
    ],
    answer: 'The model underfits and fails to capture important patterns',
    explanation:
      'High bias indicates underfitting, where the model oversimplifies the problem and fails to capture significant patterns.',
  },
  {
    id: 65,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'Which of the following is a solution for addressing high bias in machine learning?',
    options: [
      'Increase model complexity, add features',
      'Simplify the model, add more data',
      'Reduce training data size',
      'Use regularization',
    ],
    answer: 'Increase model complexity, add features',
    explanation:
      'To address high bias, increasing model complexity or adding more features can help the model capture more patterns.',
  },
  {
    id: 66,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'What is the primary cause of high variance in a machine learning model?',
    options: [
      'The model captures noise along with patterns',
      'The model fails to capture important patterns',
      'The model uses a small training dataset',
      'The model complexity is too low',
    ],
    answer: 'The model captures noise along with patterns',
    explanation:
      'High variance indicates overfitting, where the model becomes overly sensitive to fluctuations in the training data.',
  },
  {
    id: 67,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'What is the main effect of high variance in a machine learning model?',
    options: [
      'The model performs excellently on training data but poorly on test data',
      'The model fails to capture any patterns in the data',
      'The model underfits and shows poor performance on both training and test data',
      'The model is insensitive to small changes in training data',
    ],
    answer:
      'The model performs excellently on training data but poorly on test data',
    explanation:
      'High variance results in overfitting, where the model performs well on training data but fails to generalize to new, unseen data.',
  },
  {
    id: 68,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'Which of the following techniques is used to reduce variance in a model?',
    options: [
      'Simplify the model, add more data, use regularization',
      'Increase model complexity',
      'Add more features to the model',
      'Use a more complex algorithm',
    ],
    answer: 'Simplify the model, add more data, use regularization',
    explanation:
      'Reducing variance typically involves simplifying the model, adding more data, or using techniques like regularization.',
  },
  {
    id: 69,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'What is the primary purpose of bagging (Bootstrap Aggregating)?',
    options: [
      'To reduce variance and improve model stability by combining multiple models',
      'To improve model performance by increasing the complexity of the base models',
      'To reduce bias by focusing on misclassified instances',
      'To create a single, more powerful model by combining various models',
    ],
    answer:
      'To reduce variance and improve model stability by combining multiple models',
    explanation:
      'Bagging aims to reduce variance by combining the predictions of multiple models trained on random subsets of the data.',
  },
  {
    id: 70,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'In the context of bagging, what is the method used to create training data subsets?',
    options: [
      'Sampling with replacement',
      'Randomly selecting a subset of features',
      'Using all available data without repetition',
      'Clustering the data before sampling',
    ],
    answer: 'Sampling with replacement',
    explanation:
      'Bagging uses sampling with replacement to create multiple random subsets of the training data for training different models.',
  },
  {
    id: 71,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'What is the final prediction method in bagging models?',
    options: [
      'Majority vote or averaging',
      'Weighted average of predictions',
      'Simple arithmetic mean',
      'Voting based on confidence scores',
    ],
    answer: 'Majority vote or averaging',
    explanation:
      'In bagging, the final prediction is made by aggregating the predictions of multiple models through majority voting (for classification) or averaging (for regression).',
  },
  {
    id: 72,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'Which of the following is a key difference between bagging and random forest?',
    options: [
      'Random Forest introduces randomness in feature selection at each split, while bagging uses all features',
      'Random Forest uses more models than bagging',
      'Bagging involves sequential model training, while Random Forest trains models in parallel',
      'Bagging selects random features at each split, whereas Random Forest uses all features',
    ],
    answer:
      'Random Forest introduces randomness in feature selection at each split, while bagging uses all features',
    explanation:
      'Random Forest further improves bagging by adding randomization in feature selection at each split to reduce correlation between trees.',
  },
  {
    id: 73,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'When is bagging most useful?',
    options: [
      'When the base model is unstable and prone to high variance',
      'When the data is linearly separable',
      'When the model needs to capture complex patterns in the data',
      'When the training data is very small',
    ],
    answer: 'When the base model is unstable and prone to high variance',
    explanation:
      'Bagging is most effective for unstable models like decision trees, as it reduces variance and improves stability.',
  },
  {
    id: 74,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'What is the primary purpose of boosting in machine learning?',
    options: [
      'To reduce bias by sequentially improving weak models based on misclassified instances',
      'To reduce variance by combining multiple models in parallel',
      'To simplify the model by removing irrelevant features',
      'To combine predictions from different algorithms',
    ],
    answer:
      'To reduce bias by sequentially improving weak models based on misclassified instances',
    explanation:
      'Boosting works by iteratively training weak learners, focusing on the instances that previous learners misclassified, which reduces bias.',
  },
  {
    id: 75,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'What is the key characteristic of boosting algorithms?',
    options: [
      'They focus on the misclassified instances by adjusting their weights in each iteration',
      'They combine models trained on different data subsets in parallel',
      'They create a single model by averaging the predictions of multiple models',
      'They rely on cross-validation to improve model accuracy',
    ],
    answer:
      'They focus on the misclassified instances by adjusting their weights in each iteration',
    explanation:
      'Boosting algorithms like AdaBoost adjust the weight of misclassified instances to focus on improving model accuracy for those samples.',
  },
  {
    id: 76,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'In stacking, how is the final prediction made?',
    options: [
      'By using a meta-model to learn from the predictions of base models',
      'By averaging the predictions of base models',
      'By choosing the best base model based on its accuracy',
      'By using the majority vote of base models',
    ],
    answer:
      'By using a meta-model to learn from the predictions of base models',
    explanation:
      'Stacking combines multiple base models and uses a meta-model to learn from their predictions to make the final prediction.',
  },
  {
    id: 77,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which parameter should you adjust if you want to control the depth of a decision tree in scikit-learn?',
    options: [
      'max_depth',
      'n_estimators',
      'learning_rate',
      'min_samples_split',
    ],
    answer: 'max_depth',
    explanation:
      'The `max_depth` parameter controls the maximum depth of the tree, preventing it from growing too deep and overfitting the data.',
  },
  {
    id: 78,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which parameter should you adjust to change the number of neighbors in a k-Nearest Neighbors (kNN) model in scikit-learn?',
    options: ['n_neighbors', 'algorithm', 'metric', 'leaf_size'],
    answer: 'n_neighbors',
    explanation:
      'The `n_neighbors` parameter defines the number of neighbors to use for classification or regression in a kNN model.',
  },
  {
    id: 79,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which parameter should be set in scikit-learn’s `LogisticRegression` to adjust the regularization strength?',
    options: ['C', 'penalty', 'solver', 'max_iter'],
    answer: 'C',
    explanation:
      'The `C` parameter controls the regularization strength in logistic regression. A smaller value means stronger regularization.',
  },
  {
    id: 80,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      "Which parameter in scikit-learn's `RandomForestClassifier` can be used to control the minimum number of samples required to split an internal node?",
    options: ['min_samples_split', 'n_estimators', 'max_features', 'max_depth'],
    answer: 'min_samples_split',
    explanation:
      'The `min_samples_split` parameter determines the minimum number of samples required to split an internal node in a decision tree.',
  },
  {
    id: 81,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      "Which parameter should you use in scikit-learn's `SVC` (Support Vector Classifier) to adjust the penalty for misclassification?",
    options: ['C', 'kernel', 'gamma', 'degree'],
    answer: 'C',
    explanation:
      'The `C` parameter in `SVC` controls the penalty for misclassification. A higher value of `C` aims to reduce misclassification but may lead to overfitting.',
  },
  {
    id: 82,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which parameter should you adjust in a `RandomForestRegressor` to control the number of trees in the forest?',
    options: ['n_estimators', 'max_depth', 'min_samples_split', 'max_features'],
    answer: 'n_estimators',
    explanation:
      "The `n_estimators` parameter controls the number of trees in the forest. Increasing this value can improve the model's accuracy.",
  },
  {
    id: 83,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      "In scikit-learn's `GradientBoostingClassifier`, which parameter controls the learning rate of the boosting process?",
    options: ['learning_rate', 'n_estimators', 'max_depth', 'subsample'],
    answer: 'learning_rate',
    explanation:
      'The `learning_rate` parameter controls the contribution of each tree to the final prediction. A lower value makes the model more robust but slower to converge.',
  },
  {
    id: 84,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      "Which parameter should you adjust in scikit-learn's `KMeans` to control the number of clusters?",
    options: ['n_clusters', 'max_iter', 'init', 'algorithm'],
    answer: 'n_clusters',
    explanation:
      'The `n_clusters` parameter specifies the number of clusters to form in the KMeans clustering algorithm.',
  },
  {
    id: 85,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which parameter in scikit-learn’s `DecisionTreeClassifier` should you adjust to set the minimum number of samples required at a leaf node?',
    options: [
      'min_samples_leaf',
      'max_features',
      'max_depth',
      'min_samples_split',
    ],
    answer: 'min_samples_leaf',
    explanation:
      'The `min_samples_leaf` parameter defines the minimum number of samples required to be at a leaf node. Increasing this value can prevent overfitting.',
  },
  {
    id: 86,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which parameter should you adjust in scikit-learn’s `LinearRegression` to include or exclude an intercept term in the model?',
    options: ['fit_intercept', 'normalize', 'copy_X', 'n_jobs'],
    answer: 'fit_intercept',
    explanation:
      'The `fit_intercept` parameter controls whether the model includes an intercept term. Setting it to `False` forces the model to pass through the origin.',
  },
  {
    id: 87,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'What is the main goal of supervised learning?',
    options: [
      'To discover hidden patterns in the data',
      'To predict outcomes based on labeled data',
      'To group similar data points together',
      'To reduce the dimensionality of the data',
    ],
    answer: 'To predict outcomes based on labeled data',
    explanation:
      'Supervised learning models are trained using labeled data to make predictions about future or unseen data.',
  },
  {
    id: 88,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'What is the purpose of dimensionality reduction?',
    options: [
      'To make the model more interpretable',
      'To improve model accuracy',
      'To reduce the complexity of the data',
      'To add more features to the dataset',
    ],
    answer: 'To reduce the complexity of the data',
    explanation:
      'Dimensionality reduction helps in simplifying the dataset by reducing the number of features, which can lead to faster computations and reduce overfitting.',
  },
  {
    id: 89,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'Which of the following is a method for normalization?',
    options: [
      'Binning',
      'Range Normalization',
      'Sampling',
      'Covariance Matrix',
    ],
    answer: 'Range Normalization',
    explanation:
      'Range normalization rescales data within a specific range, typically [0,1], to ensure that features are on a similar scale.',
  },
  {
    id: 90,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'What does PCA do to the data?',
    options: [
      'Reduces the number of classes in the data',
      'Maximizes the class separability',
      'Reduces the dimensionality of the data while preserving variance',
      'Transforms data into a non-linear space',
    ],
    answer: 'Reduces the dimensionality of the data while preserving variance',
    explanation:
      'Principal Component Analysis (PCA) reduces the number of features in the data by creating new features (principal components) that capture the most variance in the data.',
  },
  {
    id: 91,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'What is the key difference between PCA and LDA?',
    options: [
      'PCA is unsupervised, while LDA is supervised',
      'PCA uses linear transformations, while LDA uses non-linear transformations',
      'PCA minimizes class separability, while LDA maximizes it',
      'PCA works with categorical data, while LDA works with continuous data',
    ],
    answer: 'PCA is unsupervised, while LDA is supervised',
    explanation:
      'PCA is an unsupervised technique for dimensionality reduction, while LDA is a supervised technique aimed at maximizing class separability.',
  },
  {
    id: 92,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: 'Which of the following is an unsupervised learning technique?',
    options: [
      'Decision Trees',
      'Logistic Regression',
      'k-Means Clustering',
      'Random Forest',
    ],
    answer: 'k-Means Clustering',
    explanation:
      'k-Means is an unsupervised learning algorithm used for clustering, where the goal is to group similar data points together without labeled outputs.',
  },
  {
    id: 93,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question: "What does 'curse of dimensionality' refer to?",
    options: [
      'The difficulty of visualizing data in higher dimensions',
      'The increase in complexity and computation time as the number of features increases',
      'The inability to apply dimensionality reduction techniques',
      'The need for more data when dealing with high-dimensional datasets',
    ],
    answer:
      'The increase in complexity and computation time as the number of features increases',
    explanation:
      'The curse of dimensionality refers to the challenges faced as the number of features grows, leading to increased computational requirements and potential overfitting.',
  },
  {
    id: 94,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'In Principal Component Analysis (PCA), how is the optimal number of principal components typically determined?',
    options: [
      'By selecting the components with the highest variance',
      'By using cross-validation',
      'By choosing the components with the smallest eigenvalues',
      'By examining the scree plot',
    ],
    answer: 'By examining the scree plot',
    explanation:
      'The scree plot helps visualize the eigenvalues and determine the optimal number of components by identifying the point where the variance explained by additional components levels off.',
  },
  {
    id: 95,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'What does feature selection do in the context of dimensionality reduction?',
    options: [
      'Transforms the data into a new feature space',
      'Reduces the number of features by selecting a subset of the original features',
      'Normalizes the data',
      'Removes duplicates and irrelevant data',
    ],
    answer:
      'Reduces the number of features by selecting a subset of the original features',
    explanation:
      'Feature selection helps to identify and keep only the most important features, reducing the complexity of the model without losing significant information.',
  },
  {
    id: 96,
    topic: [TopicEnum.AML, TopicEnum.DataPreprocessing],
    question:
      'Which of the following is NOT a technique used in data preprocessing?',
    options: [
      'Data Transformation',
      'Dimensionality Reduction',
      'Data Integration',
      'Random Forest',
    ],
    answer: 'Random Forest',
    explanation:
      'Random Forest is a machine learning algorithm, not a data preprocessing technique. Preprocessing techniques include transformation, integration, and reduction.',
  },
  {
    id: 97,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question:
      'Which of the following is NOT a component of time series decomposition?',
    options: ['Trend', 'Seasonality', 'Residual', 'Variance'],
    answer: 'Variance',
    explanation:
      'The three main components of time series decomposition are trend, seasonality, and residuals. Variance is a statistical property, not a decomposition component.',
  },
  {
    id: 98,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question: 'What does the seasonal component in a time series represent?',
    options: [
      'The overall direction in the data',
      'Random fluctuations',
      'Patterns that repeat at regular intervals',
      'Sudden spikes due to anomalies',
    ],
    answer: 'Patterns that repeat at regular intervals',
    explanation:
      'The seasonal component captures periodic patterns that occur at regular intervals such as months or quarters in a time series.',
  },
  {
    id: 99,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question:
      'What is the primary purpose of the TimeseriesGenerator in Keras?',
    options: [
      'Visualizing time series data',
      'Cleaning noisy time series data',
      'Preparing time series data for training',
      'Scaling time series data',
    ],
    answer: 'Preparing time series data for training',
    explanation:
      "Keras' TimeseriesGenerator is used to efficiently prepare batches of sequential data for training models like RNNs.",
  },
  {
    id: 100,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question:
      'Which real-world application is NOT a typical example of sequence data usage?',
    options: [
      'Speech recognition',
      'DNA sequence analysis',
      'Image classification',
      'Machine translation',
    ],
    answer: 'Image classification',
    explanation:
      'Image classification is generally a static input task, while sequence data tasks involve inputs and/or outputs that are ordered or time-dependent.',
  },
  {
    id: 101,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question: 'What makes RNNs particularly suitable for time series data?',
    options: [
      'They use attention mechanisms',
      'They operate in parallel across time steps',
      'They retain memory of previous steps in the sequence',
      'They require less data for training',
    ],
    answer: 'They retain memory of previous steps in the sequence',
    explanation:
      'RNNs maintain a hidden state that carries information from previous time steps, making them effective for sequential data like time series.',
  },
  {
    id: 102,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question:
      'Which neural network architecture handles long-term dependencies better than standard RNNs?',
    options: [
      'Feed-forward network',
      'Sequential dense layers',
      'Basic RNN',
      'LSTM',
    ],
    answer: 'LSTM',
    explanation:
      'LSTMs (Long Short-Term Memory) are designed to address the vanishing gradient problem in standard RNNs and can maintain long-term dependencies.',
  },
  {
    id: 103,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question:
      'Which of the following is an example of a Many-to-One RNN model?',
    options: [
      'Image captioning',
      'Language translation',
      'Speech recognition',
      'Sentiment analysis',
    ],
    answer: 'Sentiment analysis',
    explanation:
      'Sentiment analysis processes a sequence of words (many inputs) and produces a single sentiment label (one output), fitting the Many-to-One structure.',
  },
  {
    id: 104,
    topic: [TopicEnum.AML, TopicEnum.TimeSeriesRNN],
    question:
      'What common problem do RNNs face when processing long sequences?',
    options: [
      'They can only predict categorical outputs',
      'They cannot be trained on batches',
      'They lose memory of earlier inputs',
      'They always overfit',
    ],
    answer: 'They lose memory of earlier inputs',
    explanation:
      'RNNs suffer from vanishing gradients, which cause them to "forget" long-term information. This is why LSTM and GRU were developed.',
  },
  {
    id: 105,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: 'In Power BI, which of the following can be used to filter data?',
    options: ['Treemap', 'Funnel', 'Slicer', 'Gauge'],
    answer: 'Slicer',
    explanation:
      'Slicers are specifically designed in Power BI to filter data interactively on reports. Other visuals like Treemap, Funnel, and Gauge are used for displaying data, not filtering.',
  },
  {
    id: 106,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which loss function can be used with a model that is to predict the house price based on 10 different attributes?',
    options: [
      'MSE',
      'Binary Cross Entropy',
      'Hinge Loss',
      'Categorical Cross Entropy',
    ],
    answer: 'MSE',
    explanation:
      'For regression tasks like predicting house prices, Mean Squared Error (MSE) is commonly used because it measures the average squared difference between predicted and actual values.',
  },
  {
    id: 107,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'Which one of the following represent the number of times the algorithm scans the entire dataset?',
    options: ['Epoch', 'Batch', 'Iteration or Epoch', 'Iteration'],
    answer: 'Epoch',
    explanation:
      'An epoch is defined as one full pass through the entire training dataset. Iterations refer to the number of batches processed, and a batch is a subset of the training data.',
  },
  {
    id: 108,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question:
      'There is a dataset with 500 images, and the images are of Cat, Dog, Horse, Mouse, Bat. Which loss function is the best to use in CNN',
    options: [
      'Categorical Cross Entropy',
      'Binary Cross Entropy',
      'Sparse Categorical Cross Entropy',
      'Hinge Loss',
    ],
    answer: 'Categorical Cross Entropy',
    explanation:
      'Categorical Cross Entropy is used when dealing with multi-class classification problems where the labels are one-hot encoded. It is ideal for CNNs working with multiple distinct categories like animal types.',
  },
  {
    id: 109,
    topic: [TopicEnum.AML, TopicEnum.ScikitLearn],
    question: 'Which loss function can be used for SVM?',
    options: ['Hinge Loss', 'Cross Entropy', 'MSE', 'MAE'],
    answer: 'Hinge Loss',
    explanation:
      'Support Vector Machines (SVM) typically use the Hinge Loss function, which helps to maximize the margin between different classes.',
  },
  {
    id: 110,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'You have 20 instances in your dataset. Which approach is good for ensemble learning?',
    options: ['Any of these', 'Bagging', 'Random Forest', 'Stacking'],
    answer: 'Stacking',
    explanation:
      'With a small dataset (e.g., 20 instances), stacking can be more effective as it combines predictions from multiple models, while methods like bagging and random forest often need larger datasets to perform well.',
  },
  {
    id: 111,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question:
      'We have a Profit column in our dataset. We can create Total Profit using:',
    options: ['New Column', 'New Attribute', 'New Measure', 'New Category'],
    answer: 'New Measure',
    explanation:
      'In data analysis tools like Power BI, a "New Measure" is used to create aggregations like Total Profit, especially when it needs to be calculated dynamically based on filters.',
  },
  {
    id: 112,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'A company is interested in building a fraud detection model. Currently, the Data Scientist does not have a sufficient amount of information due to the low number of fraud cases. Which method is MOST likely to detect the GREATEST number of valid fraud cases?',
    options: [
      'Oversampling using bootstrapping',
      'Undersampling',
      'Oversampling using SMOTE',
      'Class weight adjustment',
    ],
    answer: 'Oversampling using SMOTE',
    explanation:
      'SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic examples for the minority class, helping to improve recall and detect more fraud cases in imbalanced datasets.',
  },
  {
    id: 113,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'You have a dataset with 80000 instances. Number of Epochs is set as 3. How many iterations will be there?',
    options: ['1875', '2500', '833', '7500'],
    answer: '2500',
    explanation:
      "Iterations = Number of Batches = (Total Instances / Batch Size) × Epochs. Assuming a default batch size of 32: 80000 / 32 = 2500 batches per epoch × 3 epochs = 7500 iterations. But if the answer is 2500, it's most likely referring to per-epoch iterations.",
  },
  {
    id: 114,
    topic: [TopicEnum.AML, TopicEnum.Visualizations],
    question: 'Which is not a visual that is available in PowerBI?',
    options: ['Stacked Bar Chart', 'Power KPI Chart', 'Donut Chart', 'Map'],
    answer: 'Power KPI Chart',
    explanation:
      'Power KPI Chart is not a default visual in Power BI. It can be imported as a custom visual, but it is not built-in like Stacked Bar Chart or Donut Chart.',
  },
  {
    id: 115,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'OCSVM developed by Tax & Duin uses the concept of',
    options: [
      'hyperplane',
      'hypersphere and a then a hyperplane within the hypersphere to separate instances by applying weights',
      'hypersphere',
      'hyperplane and a then hypersphere in the already separate region with the hyperplane',
    ],
    answer: 'hypersphere',
    explanation:
      'The original One-Class SVM by Tax & Duin is based on the concept of enclosing data in a minimum volume hypersphere to detect outliers or novelty instances.',
  },
  {
    id: 116,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'OCSVM creates a hyperplane between',
    options: [
      'instances and the origin',
      'class 1 and outliers of class 2',
      'class 1 and class 2',
      'class 2 and outliers of class 1',
    ],
    answer: 'instances and the origin',
    explanation:
      'In OCSVM, the goal is to find a decision boundary that separates the data from the origin, especially in high-dimensional feature space.',
  },
  {
    id: 117,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'Which method uses weak stable classifiers?',
    options: ['Bagging', 'Majority Vote', 'All of these', 'Boosting'],
    answer: 'Boosting',
    explanation:
      'Boosting uses weak learners (often decision stumps) and focuses on improving them sequentially to form a strong ensemble classifier.',
  },
  {
    id: 118,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'Bagging is',
    options: [
      'soosted Assomerave grouping',
      'Bootstrap Aggregating',
      'Balanced Aggregating',
      'Bound Agglomerative grouping',
    ],
    answer: 'Bootstrap Aggregating',
    explanation:
      'Bagging stands for Bootstrap Aggregating, a technique to reduce variance by training on multiple bootstrapped subsets and averaging the predictions.',
  },
  {
    id: 119,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'Which of the following method reduces the effect of high variance in data?',
    options: ['Boosting', 'Majority vote', 'Bagging', 'All of these'],
    answer: 'Bagging',
    explanation:
      'Bagging helps reduce variance by training each model on different subsets of data and averaging the results, thereby reducing overfitting.',
  },
  {
    id: 120,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question: 'Random Forest uses',
    options: ['Bagging', 'Boosting', 'Both of these', 'Any of these'],
    answer: 'Bagging',
    explanation:
      'Random Forest is an ensemble learning method that builds multiple decision trees using the Bagging technique.',
  },
  {
    id: 121,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'OCSVM that creates a hyperplane',
    options: [
      'which is circular in shape',
      'needs an upper bound on the number of outliers',
      'minimizes the volume that is to the right of the hyperplane',
      'uses soft margin for smoothness',
    ],
    answer: 'needs an upper bound on the number of outliers',
    explanation:
      'OCSVM requires an upper bound (nu parameter) to control the fraction of outliers and the number of support vectors.',
  },
  {
    id: 122,
    topic: [TopicEnum.AML],
    question: 'The harmonic mean of precision and recall is',
    options: ['F1 score', 'Accuracy', 'Specificity', 'Focal Loss'],
    answer: 'F1 score',
    explanation:
      'F1 score is the harmonic mean of precision and recall, providing a balance between the two in evaluating classification models.',
  },
  {
    id: 123,
    topic: [TopicEnum.AML],
    question: 'SMOTE is',
    options: [
      'Sampled Majority Oversampling Technique',
      'Synthetic Majority Oversampling Technique',
      'Synthetic Minority Oversampling Technique',
      'Sampled Minority Oversampling Technique',
    ],
    answer: 'Synthetic Minority Oversampling Technique',
    explanation:
      'SMOTE stands for Synthetic Minority Oversampling Technique. It generates synthetic samples for the minority class to balance the dataset.',
  },
  {
    id: 124,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question: 'OCSVM can be used for',
    options: [
      'Outlier Detection',
      'Regression',
      'Classification',
      'All of these',
    ],
    answer: 'Outlier Detection',
    explanation:
      'OCSVM is commonly used for outlier or anomaly detection where the model is trained only on normal instances to identify unusual patterns.',
  },
  {
    id: 125,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'What is the primary advantage of PyTorch’s dynamic computation graph?',
    options: [
      'It allows models to be exported to other platforms easily',
      'It enables on-the-fly graph modifications during runtime',
      'It reduces GPU memory usage automatically',
      'It ensures backward compatibility with Lua code',
    ],
    answer: 'It enables on-the-fly graph modifications during runtime',
    explanation:
      'Dynamic computation graphs allow you to define and modify the computational graph at runtime, which makes debugging and experimentation easier and more flexible.',
  },
  {
    id: 126,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which of the following PyTorch components is responsible for automatic gradient computation?',
    options: ['Tensor', 'DataLoader', 'Autograd', 'Optimizer'],
    answer: 'Autograd',
    explanation:
      'Autograd is PyTorch’s automatic differentiation engine that tracks operations on tensors and automatically computes gradients for backward propagation.',
  },
  {
    id: 127,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'In PyTorch, what is the main purpose of the `DataLoader` class?',
    options: [
      'To create neural network layers',
      'To apply activation functions',
      'To load and batch data efficiently during training',
      'To optimize model parameters',
    ],
    answer: 'To load and batch data efficiently during training',
    explanation:
      'The DataLoader class helps load data in batches, shuffle data, and apply multiprocessing for faster and more efficient data pipeline during training.',
  },
  {
    id: 128,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which PyTorch module contains pre-built layers and functions for building neural networks?',
    options: ['torch.optim', 'torch.nn', 'torch.utils', 'torch.cuda'],
    answer: 'torch.nn',
    explanation:
      'torch.nn is the core module in PyTorch for building neural networks. It includes commonly used layers like Linear, Conv2d, activation functions, and more.',
  },
  {
    id: 129,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which optimizer in PyTorch is typically preferred for its adaptive learning rate features?',
    options: ['SGD', 'Adam', 'RMSprop', 'Nesterov'],
    answer: 'Adam',
    explanation:
      'Adam combines the advantages of both RMSprop and SGD with momentum. It adapts learning rates based on the first and second moments of gradients, making it efficient for most applications.',
  },
  {
    id: 130,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'What is the function of the `forward()` method in a PyTorch model?',
    options: [
      'It initializes the model’s parameters',
      'It performs a backward pass for gradient calculation',
      'It defines how input data flows through the network layers',
      'It shuffles the training data',
    ],
    answer: 'It defines how input data flows through the network layers',
    explanation:
      'The forward() method in a subclass of `nn.Module` defines the computation performed at every call and determines how the data flows through the network.',
  },
  {
    id: 131,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'What does `requires_grad=True` do when assigned to a tensor in PyTorch?',
    options: [
      'Prevents the tensor from being modified',
      'Marks the tensor as a constant',
      'Enables PyTorch to track all operations on the tensor for autograd',
      'Optimizes the tensor using torch.optim',
    ],
    answer:
      'Enables PyTorch to track all operations on the tensor for autograd',
    explanation:
      'Setting `requires_grad=True` lets PyTorch keep track of all operations on the tensor so that it can automatically compute gradients during backpropagation.',
  },
  {
    id: 132,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'Which method is used to move a PyTorch tensor to the GPU?',
    options: ['.to(cpu)', '.cuda()', '.numpy()', '.device("gpu")'],
    answer: '.cuda()',
    explanation:
      'Calling `.cuda()` on a tensor moves it to the default GPU device for accelerated computation.',
  },
  {
    id: 133,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'What is `TorchScript` used for in PyTorch?',
    options: [
      'To visualize model training metrics',
      'To write models using C++ only',
      'To optimize and serialize models for production deployment',
      'To debug Python scripts using graphs',
    ],
    answer: 'To optimize and serialize models for production deployment',
    explanation:
      'TorchScript is an intermediate representation of a PyTorch model that can be optimized and run independently from Python, useful for deployment in production environments.',
  },
  {
    id: 134,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which of the following best describes the `nn.Module` class in PyTorch?',
    options: [
      'A utility class for loading data',
      'A base class for all neural network models',
      'A special type of tensor with gradients',
      'A module that contains GPU drivers',
    ],
    answer: 'A base class for all neural network models',
    explanation:
      '`nn.Module` is the base class for all neural network models in PyTorch. It encapsulates parameters, layers, and the forward pass logic.',
  },
  {
    id: 135,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question: 'What is the main goal of object detection in computer vision?',
    options: [
      'Classify an image into a category',
      'Segment an object at the pixel level',
      'Locate and classify multiple objects within an image',
      'Generate text descriptions of an image',
    ],
    answer: 'Locate and classify multiple objects within an image',
    explanation:
      'Object detection involves identifying instances of objects and determining their locations in an image using bounding boxes.',
  },
  {
    id: 136,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which of the following best describes a "bounding box" in object detection?',
    options: [
      'A mask that outlines the object’s shape',
      'A coordinate that marks the center of the object',
      'A rectangle that encloses an object and defines its position',
      'A 3D cube that models the object’s depth',
    ],
    answer: 'A rectangle that encloses an object and defines its position',
    explanation:
      'A bounding box is a rectangular box used in object detection to describe the location of an object within an image.',
  },
  {
    id: 137,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What does the "IoU" (Intersection over Union) metric measure in object detection?',
    options: [
      'The number of detected objects',
      'The speed of model inference',
      'The overlap between predicted and ground truth bounding boxes',
      'The classification accuracy of objects',
    ],
    answer: 'The overlap between predicted and ground truth bounding boxes',
    explanation:
      'IoU measures how much the predicted bounding box overlaps with the ground truth box. It is used to evaluate localization accuracy.',
  },
  {
    id: 138,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which type of neural network architecture is most commonly used for object detection?',
    options: [
      'Recurrent Neural Networks',
      'Convolutional Neural Networks',
      'Fully Connected Networks',
      'Generative Adversarial Networks',
    ],
    answer: 'Convolutional Neural Networks',
    explanation:
      'CNNs are the backbone of most object detection models due to their effectiveness in extracting spatial features from images.',
  },
  {
    id: 139,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'In object detection, what is the purpose of Non-Maximum Suppression (NMS)?',
    options: [
      'To convert object masks into bounding boxes',
      'To calculate the classification score',
      'To remove duplicate detections of the same object',
      'To resize images before inputting to the model',
    ],
    answer: 'To remove duplicate detections of the same object',
    explanation:
      'NMS filters out overlapping bounding boxes by keeping only the one with the highest confidence score for a given object.',
  },
  {
    id: 140,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which of the following best describes a key difference between SSD and YOLO?',
    options: [
      'SSD processes images sequentially while YOLO processes in parallel',
      'YOLO performs object detection as a regression problem, while SSD uses a two-stage approach',
      'SSD uses predefined anchor boxes at multiple feature map scales, while YOLO predicts boxes directly from the image',
      'YOLO generates masks for each object, SSD does not',
    ],
    answer:
      'SSD uses predefined anchor boxes at multiple feature map scales, while YOLO predicts boxes directly from the image',
    explanation:
      'SSD uses anchor boxes on multiple feature maps to detect objects at different scales, while YOLO predicts bounding boxes directly from the image grid.',
  },
  {
    id: 141,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which object detection model is generally faster and optimized for real-time detection?',
    options: ['SSD', 'YOLO', 'Faster R-CNN', 'Mask R-CNN'],
    answer: 'YOLO',
    explanation:
      'YOLO (You Only Look Once) is designed for speed and real-time detection by predicting all bounding boxes and class probabilities in a single evaluation.',
  },
  {
    id: 142,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which of the following best describes how YOLO divides an image for object detection?',
    options: [
      'By sliding window over the image',
      'By generating region proposals using selective search',
      'By splitting the image into a grid and predicting boxes per grid cell',
      'By detecting objects on multiple feature map levels',
    ],
    answer:
      'By splitting the image into a grid and predicting boxes per grid cell',
    explanation:
      'YOLO divides the input image into an SxS grid, and each grid cell predicts bounding boxes and class probabilities.',
  },
  {
    id: 143,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question: 'What is one advantage of SSD over YOLO in object detection?',
    options: [
      'SSD can detect objects without training',
      'SSD uses attention mechanisms to locate objects',
      'SSD handles objects of varying sizes better due to multi-scale feature maps',
      'SSD is a two-stage detector, making it more accurate than YOLO',
    ],
    answer:
      'SSD handles objects of varying sizes better due to multi-scale feature maps',
    explanation:
      'SSD applies detection at multiple feature map levels, making it better at detecting objects of different sizes compared to early versions of YOLO.',
  },
  {
    id: 144,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which version of YOLO introduced anchor boxes to improve object detection performance?',
    options: ['YOLOv1', 'YOLOv2', 'YOLOv3', 'YOLOv4'],
    answer: 'YOLOv2',
    explanation:
      'YOLOv2 introduced anchor boxes to allow better localization and improve accuracy, a concept previously used in SSD.',
  },
  {
    id: 145,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which metric is commonly used to evaluate object detection models?',
    options: [
      'Accuracy',
      'F1 Score',
      'Mean Average Precision (mAP)',
      'Root Mean Square Error (RMSE)',
    ],
    answer: 'Mean Average Precision (mAP)',
    explanation:
      'mAP evaluates the precision-recall curve for each class and then averages them, providing a comprehensive measure of detection performance.',
  },
  {
    id: 146,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'During training, what is the purpose of non-maximum suppression (NMS)?',
    options: [
      'To remove irrelevant labels from training data',
      'To smooth bounding box edges',
      'To reduce duplicate predictions by keeping the highest scoring box',
      'To normalize image pixel values',
    ],
    answer:
      'To reduce duplicate predictions by keeping the highest scoring box',
    explanation:
      'NMS eliminates redundant bounding boxes by keeping the one with the highest confidence score and suppressing others that overlap significantly.',
  },
  {
    id: 147,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which technique is commonly used to make object detection models more robust to different image conditions?',
    options: [
      'Backpropagation',
      'Data augmentation',
      'Gradient clipping',
      'Dropout',
    ],
    answer: 'Data augmentation',
    explanation:
      'Data augmentation techniques like flipping, cropping, and color jittering increase variability in training data, making the model more robust.',
  },
  {
    id: 148,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Why is it important to use a validation set during training of object detection models?',
    options: [
      'To reduce the size of the training set',
      'To fine-tune labels',
      'To evaluate generalization performance and prevent overfitting',
      'To speed up training time',
    ],
    answer: 'To evaluate generalization performance and prevent overfitting',
    explanation:
      'A validation set helps monitor model performance on unseen data during training, ensuring it generalizes well beyond the training set.',
  },
  {
    id: 149,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'In an object detection architecture, what is the primary role of the detection head?',
    options: [
      'To resize input images',
      'To extract low-level features',
      'To predict class probabilities and bounding box coordinates',
      'To perform batch normalization',
    ],
    answer: 'To predict class probabilities and bounding box coordinates',
    explanation:
      'The detection head takes the feature maps from the backbone or neck and outputs object classes and bounding box coordinates.',
  },
  {
    id: 150,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What makes detection heads in YOLO and SSD different from those in Faster R-CNN?',
    options: [
      'YOLO and SSD use a two-stage detection approach',
      'YOLO and SSD predict detections directly from feature maps in a single stage',
      'Faster R-CNN skips region proposals',
      'YOLO and SSD do not use convolutional layers',
    ],
    answer:
      'YOLO and SSD predict detections directly from feature maps in a single stage',
    explanation:
      'YOLO and SSD are single-stage detectors that skip region proposal generation and make direct predictions, unlike the two-stage Faster R-CNN.',
  },
  {
    id: 151,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which of the following best describes the transition from classical to modern object detectors?',
    options: [
      'From pixel-by-pixel classification to anchor-based regression and classification',
      'From anchor-based to region-based object classification',
      'From segmentation masks to grayscale regression',
      'From CNNs to k-NN classifiers',
    ],
    answer:
      'From pixel-by-pixel classification to anchor-based regression and classification',
    explanation:
      'Modern detectors like SSD and YOLO moved toward using anchors and directly regressing box coordinates with classification scores.',
  },
  {
    id: 152,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What is a major challenge when using anchor boxes in object detection models?',
    options: [
      'They increase the resolution of input images',
      'They limit model inference speed',
      'They require careful tuning of scales and aspect ratios',
      'They reduce the number of trainable parameters',
    ],
    answer: 'They require careful tuning of scales and aspect ratios',
    explanation:
      'Anchor boxes need to be well-matched to the dataset’s object sizes and shapes, which often requires manual tuning or clustering techniques.',
  },
  {
    id: 153,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What key feature distinguishes Transformer-based detectors like DETR from traditional convolutional-based detectors?',
    options: [
      'Use of RNNs for spatial reasoning',
      'Anchor-free, end-to-end prediction with global attention',
      'Single-class detection only',
      'Use of 3D convolutions for every layer',
    ],
    answer: 'Anchor-free, end-to-end prediction with global attention',
    explanation:
      'DETR uses the Transformer architecture to model global dependencies in an anchor-free way, predicting boxes and classes directly.',
  },
  {
    id: 154,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What is the main purpose of the Region Proposal Network (RPN) in Faster R-CNN?',
    options: [
      'To classify final object classes',
      'To apply non-maximum suppression',
      'To generate candidate object regions for further classification',
      'To resize the input image',
    ],
    answer: 'To generate candidate object regions for further classification',
    explanation:
      'The RPN suggests regions in the image likely to contain objects, which are then classified and refined by the detection head.',
  },
  {
    id: 155,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which traditional object detection model introduced the concept of anchor boxes?',
    options: ['YOLOv1', 'SSD', 'Faster R-CNN', 'R-CNN'],
    answer: 'Faster R-CNN',
    explanation:
      'Faster R-CNN introduced the use of anchor boxes in the Region Proposal Network to suggest potential object locations.',
  },
  {
    id: 156,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question: 'What is the main drawback of the original R-CNN approach?',
    options: [
      'It lacks classification capability',
      'It requires a large pre-trained transformer model',
      'It uses very slow region-wise feature extraction',
      'It does not work on grayscale images',
    ],
    answer: 'It uses very slow region-wise feature extraction',
    explanation:
      'R-CNN extracts CNN features for each region proposal individually, which makes it very slow compared to later methods like Fast R-CNN.',
  },
  {
    id: 157,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question: 'How does Fast R-CNN improve over the original R-CNN?',
    options: [
      'It replaces CNNs with RNNs',
      'It runs the CNN over the entire image only once and uses RoI pooling',
      'It removes the need for classification',
      'It eliminates the need for bounding box regression',
    ],
    answer:
      'It runs the CNN over the entire image only once and uses RoI pooling',
    explanation:
      'Fast R-CNN significantly reduces computation time by extracting features from the whole image and then pooling regions of interest (RoIs).',
  },
  {
    id: 158,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'In traditional object detectors, what is the role of Non-Maximum Suppression (NMS)?',
    options: [
      'To increase image resolution',
      'To combine object classes',
      'To remove duplicate detections by keeping the most confident one',
      'To normalize bounding box coordinates',
    ],
    answer: 'To remove duplicate detections by keeping the most confident one',
    explanation:
      'NMS is used to suppress overlapping boxes that predict the same object, keeping only the one with the highest confidence score.',
  },
  {
    id: 159,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What is the primary purpose of the sliding window technique in traditional object detection?',
    options: [
      'To generate synthetic training images',
      'To detect edges at different scales',
      'To scan the image for objects at various positions and scales',
      'To adjust the resolution of feature maps',
    ],
    answer: 'To scan the image for objects at various positions and scales',
    explanation:
      'The sliding window technique systematically scans the image across positions and scales to detect objects using classifiers.',
  },
  {
    id: 160,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Which of the following best describes the role of SIFT in traditional object detection?',
    options: [
      'To classify full images into categories',
      'To detect keypoints that are invariant to scale and rotation',
      'To aggregate pixel intensities into a histogram',
      'To reduce dimensionality of the input image',
    ],
    answer: 'To detect keypoints that are invariant to scale and rotation',
    explanation:
      'SIFT is used to extract local features that are robust to changes in scale, rotation, and illumination.',
  },
  {
    id: 161,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'What is a key characteristic of HOG (Histogram of Oriented Gradients)?',
    options: [
      'It compresses images using gradient descent',
      'It detects image scale using keypoints',
      'It captures object shape by computing gradient orientation histograms',
      'It generates synthetic gradients for GANs',
    ],
    answer:
      'It captures object shape by computing gradient orientation histograms',
    explanation:
      'HOG captures the structure of an object by analyzing the distribution of edge directions or gradients.',
  },
  {
    id: 162,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'In traditional object detection pipelines, what is the role of Support Vector Machines (SVMs)?',
    options: [
      'To generate region proposals',
      'To fine-tune convolutional filters',
      'To classify sliding window patches as object or background',
      'To perform unsupervised clustering of features',
    ],
    answer: 'To classify sliding window patches as object or background',
    explanation:
      'SVMs were commonly used in traditional pipelines to determine whether a region contains an object based on extracted features like HOG.',
  },
  {
    id: 163,
    topic: [TopicEnum.MV, TopicEnum.ObjectDetection],
    question:
      'Why was the sliding window approach computationally expensive in traditional object detection?',
    options: [
      'Because it used deep learning models',
      'Because it required training multiple SVMs simultaneously',
      'Because it had to evaluate the classifier on many overlapping windows at various scales',
      'Because it processed all features in the frequency domain',
    ],
    answer:
      'Because it had to evaluate the classifier on many overlapping windows at various scales',
    explanation:
      'The sliding window method is computationally expensive due to the exhaustive nature of scanning over all positions and scales.',
  },
  {
    id: 164,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'What is a major limitation of traditional object tracking algorithms in motion vision?',
    options: [
      'They are unable to handle occlusions effectively',
      'They require too much memory to store tracking data',
      'They can only track one object at a time',
      'They are unable to process video frames in real-time',
    ],
    answer: 'They are unable to handle occlusions effectively',
    explanation:
      'Traditional object tracking algorithms often struggle with handling occlusions, where objects may temporarily disappear from view due to overlapping or obstructions.',
  },
  {
    id: 165,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'Which method is commonly used to overcome occlusions in object tracking?',
    options: [
      'Optical flow',
      'Kalman filter',
      'Mean-Shift tracking',
      'Template matching',
    ],
    answer: 'Kalman filter',
    explanation:
      "The Kalman filter is widely used for predicting object locations, helping to maintain tracking even during partial occlusions by predicting the object's next position.",
  },
  {
    id: 166,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'Which of the following is true about feature-based object tracking?',
    options: [
      'It uses the raw pixel values of the object for tracking',
      'It tracks the object based on selected features such as corners or edges',
      'It is computationally more expensive than model-based tracking',
      'It only works with grayscale images',
    ],
    answer:
      'It tracks the object based on selected features such as corners or edges',
    explanation:
      'Feature-based tracking relies on identifying distinct features (like corners or edges) and tracking those over time, making it more robust in dynamic environments.',
  },
  {
    id: 167,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question: 'What is the role of the optical flow method in object tracking?',
    options: [
      'It estimates the movement of objects by analyzing pixel intensity changes',
      'It tracks objects by comparing their size over time',
      'It creates a 3D model of the tracked object',
      'It matches templates of the object across frames',
    ],
    answer:
      'It estimates the movement of objects by analyzing pixel intensity changes',
    explanation:
      'Optical flow computes the motion of objects by analyzing the change in pixel intensity across successive frames, allowing for the tracking of their movement.',
  },
  {
    id: 168,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'Which algorithm is often used in real-time object tracking due to its efficiency?',
    options: [
      'Support Vector Machines (SVM)',
      'Mean-Shift',
      'K-Nearest Neighbors (KNN)',
      'Convolutional Neural Networks (CNN)',
    ],
    answer: 'Mean-Shift',
    explanation:
      'The Mean-Shift algorithm is known for its efficiency and ability to track objects in real-time by searching for the mode of the color histogram in each frame.',
  },
  {
    id: 169,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'What challenge is common when tracking multiple objects in a video sequence?',
    options: [
      'Lack of sufficient computational power',
      'Difficulty in differentiating objects with similar appearance',
      'The inability to detect motion in low-light environments',
      'The inability to track objects in fast-moving sequences',
    ],
    answer: 'Difficulty in differentiating objects with similar appearance',
    explanation:
      'When multiple objects have similar appearance, it becomes difficult to differentiate and track them effectively, leading to confusion and tracking errors.',
  },
  {
    id: 170,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'What is a major limitation of template matching in object tracking?',
    options: [
      'It is highly sensitive to changes in object appearance and scale',
      'It works only with binary images',
      'It requires deep learning models for high accuracy',
      'It is only applicable to grayscale images',
    ],
    answer: 'It is highly sensitive to changes in object appearance and scale',
    explanation:
      'Template matching relies on a fixed template of the object, making it sensitive to changes in scale, rotation, or lighting conditions.',
  },
  {
    id: 171,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'Which of the following methods can be used to track objects across frames in a video sequence with varying illumination?',
    options: [
      'Optical flow',
      'Kalman filter',
      'Mean-Shift',
      'Histogram-based tracking',
    ],
    answer: 'Histogram-based tracking',
    explanation:
      'Histogram-based tracking is more robust to varying illumination as it tracks the object based on its color distribution, which is less affected by lighting changes.',
  },
  {
    id: 172,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'In object tracking, what does the Kalman filter help to predict?',
    options: [
      "The object's exact location in each frame",
      'The movement of the object based on a probabilistic model',
      "The object's appearance over time",
      "The object's velocity and acceleration",
    ],
    answer: 'The movement of the object based on a probabilistic model',
    explanation:
      'The Kalman filter predicts the state of an object (position and velocity) based on prior states and measurements, accounting for uncertainty in the system.',
  },
  {
    id: 173,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'Which object tracking technique is known for being computationally efficient, particularly in real-time applications?',
    options: [
      'Hough transform',
      'Mean-Shift',
      'Optical flow',
      'Particle filter',
    ],
    answer: 'Mean-Shift',
    explanation:
      'The Mean-Shift algorithm is computationally efficient, which makes it suitable for real-time object tracking, especially in dynamic environments.',
  },
  {
    id: 174,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question: 'Why is background subtraction commonly used in object tracking?',
    options: [
      'To segment moving objects from a static background',
      "To identify the object's color and texture",
      'To predict the trajectory of moving objects',
      'To match objects to predefined templates',
    ],
    answer: 'To segment moving objects from a static background',
    explanation:
      'Background subtraction helps to separate moving objects from a stationary background, making it easier to track the objects over time.',
  },
  {
    id: 175,
    topic: [TopicEnum.MV, TopicEnum.ObjectTracking],
    question:
      'What is the main advantage of using a particle filter in object tracking?',
    options: [
      'It can track objects in 3D space',
      'It is highly accurate for occluded objects',
      'It can track non-linear and non-Gaussian processes',
      'It is computationally faster than other methods',
    ],
    answer: 'It can track non-linear and non-Gaussian processes',
    explanation:
      'Particle filters are ideal for tracking objects in complex environments because they can handle non-linear and non-Gaussian models, making them more flexible than traditional filters.',
  },
  {
    id: 176,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'What is the primary benefit of sensor fusion in autonomous vehicles?',
    options: [
      'It reduces the need for real-time data processing',
      'It improves vehicle accuracy by combining data from multiple sensors',
      'It allows vehicles to operate without any sensors',
      'It decreases the power consumption of sensors',
    ],
    answer:
      'It improves vehicle accuracy by combining data from multiple sensors',
    explanation:
      'Sensor fusion allows autonomous vehicles to combine data from different sensors (e.g., cameras, LiDAR, radar) to improve the overall accuracy and robustness of perception, providing a more comprehensive understanding of the environment.',
  },
  {
    id: 177,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'Which sensor is most commonly used for detecting the distance to obstacles in autonomous vehicles?',
    options: ['Radar', 'LiDAR', 'Camera', 'Ultrasonic sensors'],
    answer: 'LiDAR',
    explanation:
      'LiDAR is commonly used in autonomous vehicles to measure the distance to obstacles by emitting laser pulses and measuring the time it takes for them to return, providing accurate distance data and detailed 3D maps of the environment.',
  },
  {
    id: 178,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'What is the role of Kalman filters in sensor fusion for autonomous systems?',
    options: [
      'To provide real-time data to sensors',
      'To estimate the state of a system based on noisy sensor data',
      'To reduce the computational load of the system',
      "To detect obstacles in the vehicle's path",
    ],
    answer: 'To estimate the state of a system based on noisy sensor data',
    explanation:
      "Kalman filters are used in sensor fusion to provide an optimal estimate of a system's state by combining multiple noisy sensor measurements, making it especially useful for applications like autonomous navigation.",
  },
  {
    id: 179,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'Which of the following is NOT a typical type of sensor used in autonomous vehicles?',
    options: [
      'Infrared sensor',
      'Radar sensor',
      'Ultrasonic sensor',
      'Goniometer sensor',
    ],
    answer: 'Goniometer sensor',
    explanation:
      'Goniometer sensors are not typically used in autonomous vehicles. Instead, common sensors include radar, LiDAR, ultrasonic, and cameras, which assist with perception, navigation, and obstacle detection.',
  },
  {
    id: 180,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'Why is the fusion of data from both LiDAR and cameras important in autonomous driving?',
    options: [
      'LiDAR provides high-resolution color images, while cameras detect distance',
      'Cameras offer detailed visual data, while LiDAR provides accurate depth measurements',
      'Cameras are faster than LiDAR in data processing',
      'LiDAR can detect obstacles in complete darkness, and cameras cannot',
    ],
    answer:
      'Cameras offer detailed visual data, while LiDAR provides accurate depth measurements',
    explanation:
      'LiDAR provides accurate distance measurements and 3D mapping of the environment, while cameras offer detailed color and texture information. Together, they help autonomous vehicles understand both the spatial and visual context of their surroundings.',
  },
  {
    id: 181,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'In which scenario would sensor fusion with multiple modalities (e.g., LiDAR, radar, and cameras) be particularly beneficial?',
    options: [
      "When the environment is well-lit and the vehicle's path is clear",
      'When the vehicle is navigating complex and dynamic environments with low visibility',
      'When the vehicle is in a parking lot with minimal obstacles',
      'When the vehicle is moving at low speeds only',
    ],
    answer:
      'When the vehicle is navigating complex and dynamic environments with low visibility',
    explanation:
      'Sensor fusion is particularly useful in environments with low visibility (e.g., fog, rain, or night conditions) or dynamic environments with rapidly changing obstacles. Combining sensors like LiDAR, radar, and cameras ensures that the vehicle has a more complete and reliable understanding of its surroundings.',
  },
  {
    id: 182,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'What is the main advantage of using a monocular camera in autonomous vehicles?',
    options: [
      'It provides depth information without additional sensors',
      'It is low-cost and easy to implement',
      'It can detect obstacles in 3D space',
      'It is immune to environmental conditions like rain and fog',
    ],
    answer: 'It is low-cost and easy to implement',
    explanation:
      'Monocular cameras are inexpensive and relatively simple to integrate into autonomous vehicles. However, they do not inherently provide depth information like LiDAR, which is why they are often used in combination with other sensors.',
  },
  {
    id: 183,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'What is the purpose of a sensor fusion algorithm in the context of autonomous driving?',
    options: [
      "To increase the sensor's resolution",
      'To combine data from multiple sensors for a more accurate and reliable understanding of the environment',
      'To replace the need for hardware sensors',
      "To optimize the vehicle's speed and fuel efficiency",
    ],
    answer:
      'To combine data from multiple sensors for a more accurate and reliable understanding of the environment',
    explanation:
      "Sensor fusion algorithms are designed to combine data from various sensors (such as cameras, LiDAR, radar, etc.) to improve the accuracy and reliability of the vehicle's perception of its environment, enabling safer and more precise decision-making.",
  },
  {
    id: 184,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      "Which sensor fusion technique is commonly used to estimate the vehicle's position and orientation in autonomous driving systems?",
    options: [
      'Kalman Filter',
      'Mean Shift Algorithm',
      'SIFT (Scale-Invariant Feature Transform)',
      'Histogram of Oriented Gradients (HOG)',
    ],
    answer: 'Kalman Filter',
    explanation:
      "Kalman filters are widely used in autonomous systems to estimate the vehicle's position and orientation by combining data from various sensors such as GPS, IMU, and wheel encoders. It helps produce accurate state estimates even in the presence of noisy measurements.",
  },
  {
    id: 185,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'What is the main challenge of using radar sensors in autonomous vehicles?',
    options: [
      'They are not affected by weather conditions',
      'They have poor resolution compared to LiDAR and cameras',
      'They cannot detect moving objects',
      'They require significant computational power',
    ],
    answer: 'They have poor resolution compared to LiDAR and cameras',
    explanation:
      'Radar sensors are excellent for detecting objects in various weather conditions and at long ranges. However, their resolution is lower compared to LiDAR and cameras, making them less effective at detecting fine details, such as small objects or lane markings.',
  },
  {
    id: 186,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question: 'How does a LiDAR sensor work in autonomous driving systems?',
    options: [
      'It emits sound waves and measures their return time to detect obstacles',
      'It uses laser light to create 3D maps of the environment by measuring reflected light',
      'It records high-definition video to provide visual context for decision-making',
      'It measures temperature variations to identify obstacles',
    ],
    answer:
      'It uses laser light to create 3D maps of the environment by measuring reflected light',
    explanation:
      'LiDAR sensors use laser beams to scan the environment and measure the time it takes for the reflected light to return. This data is used to create detailed 3D maps of the surroundings, helping the vehicle understand the environment in real-time.',
  },
  {
    id: 187,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'Which sensor is most effective for detecting small, stationary obstacles in the path of an autonomous vehicle?',
    options: ['Radar', 'LiDAR', 'Camera', 'Ultrasonic sensors'],
    answer: 'LiDAR',
    explanation:
      'LiDAR is particularly effective at detecting small, stationary obstacles because it provides high-resolution, accurate depth measurements. This makes it ideal for detecting objects that may be close to the vehicle, such as curbs or small debris.',
  },
  {
    id: 188,
    topic: [TopicEnum.MV, TopicEnum.SensorFusion],
    question:
      'Why is it important to use multiple sensor modalities (e.g., LiDAR, radar, cameras) in autonomous systems?',
    options: [
      'To ensure data from one sensor can compensate for the limitations of another',
      'To improve the speed of the vehicle',
      'To reduce the computational complexity of the system',
      'To allow the vehicle to drive in more traffic',
    ],
    answer:
      'To ensure data from one sensor can compensate for the limitations of another',
    explanation:
      'Each sensor modality has strengths and weaknesses. For example, radar works well in poor visibility, while cameras provide detailed visual data. By combining data from multiple sensors, an autonomous vehicle can make more informed decisions and operate more safely in diverse conditions.',
  },
  {
    id: 189,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'What does the torchvision.transforms module provide?',
    options: [
      'Serialization utilities for PyTorch models',
      'Network optimization algorithms',
      'Data encryption methods',
      'Pre-processing tools for image data',
    ],
    answer: 'Pre-processing tools for image data',
    explanation:
      'The torchvision.transforms module provides various pre-processing tools for image data, such as resizing, normalization, and augmentation, which are essential for preparing data before feeding it into a neural network.',
  },
  {
    id: 190,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'How can you convert a PyTorch Tensor to a NumPy array?',
    options: [
      'tensor.to_numpy()',
      'tensor.numpy()',
      'numpy.asarray(tensor)',
      'numpy.convert(tensor)',
    ],
    answer: 'tensor.numpy()',
    explanation:
      'The method tensor.numpy() is used to convert a PyTorch Tensor into a NumPy array. This allows for interoperability between PyTorch and other libraries like NumPy.',
  },
  {
    id: 191,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: "What is the benefit of using PyTorch's DataLoader?",
    options: [
      'GPU acceleration for data preprocessing',
      'Real-time data augmentation',
      'Automatic conversion of Python lists to tensors',
      'Efficient data loading for large datasets with easy batching and shuffling',
    ],
    answer:
      'Efficient data loading for large datasets with easy batching and shuffling',
    explanation:
      "PyTorch's DataLoader is designed to handle large datasets efficiently by automatically batching, shuffling, and loading data in parallel, making it easier to train models on large-scale data.",
  },
  {
    id: 192,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'Which package is used for building neural networks in PyTorch?',
    options: ['scipy', 'torch.nn', 'pandas', 'numpy'],
    answer: 'torch.nn',
    explanation:
      'The torch.nn package in PyTorch provides modules and classes for building neural networks. It includes layers, loss functions, and optimizers necessary for creating deep learning models.',
  },
  {
    id: 193,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: 'In PyTorch, what is a Tensor?',
    options: [
      'A type of neural network',
      'A multi-dimensional array',
      'A GPU acceleration tool',
      'A Python library for data manipulation',
    ],
    answer: 'A multi-dimensional array',
    explanation:
      'In PyTorch, a Tensor is a multi-dimensional array that is similar to a NumPy array but can be operated on GPUs for faster computation. It is the fundamental building block for working with data in PyTorch.',
  },
  {
    id: 194,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question: 'What is Machine Vision?',
    options: [
      'A branch of Artificial Intelligence focused on creating intelligent machines',
      'A technology that allows computers to interpret and process visual information',
      'A method for detecting anomalies in video streams',
      'A system for simulating human vision',
    ],
    answer:
      'A technology that allows computers to interpret and process visual information',
    explanation:
      'Machine Vision refers to the technology that enables computers to interpret and process visual data from the world, such as images or video, to perform tasks like recognition and analysis.',
  },
  {
    id: 195,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question:
      'Which of the following is a common application of Machine Vision?',
    options: [
      'Self-driving cars',
      'Facial recognition systems',
      'Industrial automation',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      'Machine Vision has diverse applications across industries, including self-driving cars, facial recognition systems, and industrial automation for quality control and defect detection.',
  },
  {
    id: 196,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question: 'What is the first step in a typical Machine Vision workflow?',
    options: [
      'Image acquisition',
      'Image preprocessing',
      'Feature extraction',
      'Pattern recognition',
    ],
    answer: 'Image acquisition',
    explanation:
      'Image acquisition is the first step in the Machine Vision workflow, where images or video streams are captured using cameras or sensors for further processing.',
  },
  {
    id: 197,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question: "In Machine Vision, what does the term 'preprocessing' refer to?",
    options: [
      'Enhancing image quality to make it suitable for analysis',
      'Detecting objects within an image',
      'Extracting key features from an image',
      'All of the above',
    ],
    answer: 'Enhancing image quality to make it suitable for analysis',
    explanation:
      'Preprocessing involves enhancing the quality of the captured image by applying techniques like filtering, noise reduction, and contrast enhancement to prepare it for analysis.',
  },
  {
    id: 198,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question:
      'Which of the following is a key component of a Machine Vision system?',
    options: [
      'A camera or image sensor',
      'A computer with software for analysis',
      'Lighting conditions for optimal image capture',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      'A typical Machine Vision system consists of a camera for image capture, software for processing and analysis, and proper lighting conditions to ensure clear image acquisition.',
  },
  {
    id: 199,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question:
      'What is the main challenge in machine vision when working with low-quality images?',
    options: [
      'Difficulty in detecting edges',
      'Overfitting the model',
      'Accuracy of object recognition',
      'Poor performance due to noisy data',
    ],
    answer: 'Poor performance due to noisy data',
    explanation:
      'Low-quality images with noise can degrade the performance of Machine Vision systems, making tasks like edge detection and object recognition more difficult.',
  },
  {
    id: 200,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question:
      'Which of the following techniques is often used in the preprocessing stage of Machine Vision?',
    options: [
      'Histogram equalization',
      'Edge detection',
      'Thresholding',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      'Techniques like histogram equalization, edge detection, and thresholding are commonly used in preprocessing to enhance the image before further analysis.',
  },
  {
    id: 201,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question: "In a Machine Vision system, what is 'feature extraction'?",
    options: [
      'Isolating specific elements of an image for further analysis',
      'Classifying the objects present in the image',
      'Adjusting the contrast and brightness of an image',
      'Detecting and removing noise from the image',
    ],
    answer: 'Isolating specific elements of an image for further analysis',
    explanation:
      'Feature extraction involves isolating important parts of an image, such as edges or textures, to help in object recognition and classification.',
  },
  {
    id: 202,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question:
      'Which industry benefits from using Machine Vision for quality control?',
    options: ['Healthcare', 'Manufacturing', 'Retail', 'All of the above'],
    answer: 'Manufacturing',
    explanation:
      'Machine Vision is widely used in manufacturing for quality control purposes, such as inspecting products for defects, measuring dimensions, and ensuring correct assembly.',
  },
  {
    id: 203,
    topic: [TopicEnum.MV, TopicEnum.MVIntro],
    question:
      'What is the main purpose of using machine vision in autonomous vehicles?',
    options: [
      'To detect and classify objects on the road',
      'To identify street signs',
      'To assist in navigation and path planning',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      'Machine Vision plays a crucial role in autonomous vehicles by detecting and classifying objects, reading street signs, and assisting in navigation and path planning.',
  },
  {
    id: 204,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question: 'Why is image processing important in Machine Vision?',
    options: [
      'It improves the performance of machine learning algorithms',
      'It helps extract useful information from images',
      'It reduces the computational cost of image analysis',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      'Image processing is important because it enhances the quality of images, reduces noise, and extracts useful features, making subsequent analysis more accurate and efficient.',
  },
  {
    id: 205,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question: 'What is the primary goal of edge detection in image processing?',
    options: [
      'To enhance image brightness',
      'To identify boundaries of objects in the image',
      'To separate the image into regions',
      'To detect faces in the image',
    ],
    answer: 'To identify boundaries of objects in the image',
    explanation:
      'Edge detection helps in identifying the boundaries of objects within an image, which is essential for tasks like object recognition and segmentation.',
  },
  {
    id: 206,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question:
      'Which image processing technique is commonly used to enhance the visibility of edges?',
    options: [
      'Canny edge detection',
      'Gaussian blur',
      'Thresholding',
      'Histogram equalization',
    ],
    answer: 'Canny edge detection',
    explanation:
      'Canny edge detection is a widely used technique that helps identify edges in an image by detecting areas of rapid intensity change.',
  },
  {
    id: 207,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question: 'What is the purpose of Gaussian blur in image processing?',
    options: [
      'To reduce noise and detail',
      'To sharpen the image',
      'To enhance contrast',
      'To detect edges',
    ],
    answer: 'To reduce noise and detail',
    explanation:
      'Gaussian blur is used to reduce noise and detail in an image by smoothing the pixel values, which helps in tasks like edge detection and segmentation.',
  },
  {
    id: 208,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question: 'What is thresholding used for in image processing?',
    options: [
      'To convert an image to binary',
      'To enhance image brightness',
      'To reduce noise',
      'To extract features from the image',
    ],
    answer: 'To convert an image to binary',
    explanation:
      'Thresholding is used to convert a grayscale image into a binary image by setting pixel values above a certain threshold to 1 (white) and below to 0 (black).',
  },
  {
    id: 209,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question:
      'What is a key benefit of using adaptive thresholding over global thresholding?',
    options: [
      'It is more sensitive to light variations in different regions of the image',
      'It is faster to compute',
      'It works well with noisy images',
      'It produces better results with large images',
    ],
    answer:
      'It is more sensitive to light variations in different regions of the image',
    explanation:
      'Adaptive thresholding adjusts the threshold dynamically for different regions of the image, making it effective for images with varying lighting conditions.',
  },
  {
    id: 210,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question:
      'Which of the following image processing techniques can be used to detect edges in an image?',
    options: [
      'Canny edge detection',
      'Histogram equalization',
      'Gaussian blur',
      'Thresholding',
    ],
    answer: 'Canny edge detection',
    explanation:
      'Canny edge detection is designed specifically to detect edges in an image by identifying areas of rapid intensity change.',
  },
  {
    id: 211,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question: 'What is the main goal of image segmentation?',
    options: [
      'To extract the most important features from an image',
      'To divide an image into regions that represent objects or parts of objects',
      'To enhance the visibility of edges',
      'To reduce image size for faster processing',
    ],
    answer:
      'To divide an image into regions that represent objects or parts of objects',
    explanation:
      'Image segmentation involves partitioning an image into distinct regions, each representing a specific object or part of an object, making it easier to analyze.',
  },
  {
    id: 212,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question:
      'Which technique is commonly used for noise removal in image processing?',
    options: [
      'Gaussian blur',
      'Sobel filter',
      'Canny edge detection',
      'Histogram equalization',
    ],
    answer: 'Gaussian blur',
    explanation:
      'Gaussian blur is commonly used to reduce noise in images, smoothing the pixel values and reducing unwanted artifacts.',
  },
  {
    id: 213,
    topic: [TopicEnum.MV, TopicEnum.ImageProcessing],
    question: 'Which of the following is NOT an image processing technique?',
    options: [
      'Edge detection',
      'Thresholding',
      'Feature extraction',
      'Data augmentation',
    ],
    answer: 'Data augmentation',
    explanation:
      'Data augmentation is not an image processing technique; it refers to artificially increasing the diversity of training data through transformations like rotations and scaling.',
  },
  {
    id: 214,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question: 'What is the purpose of image segmentation?',
    options: [
      'To identify objects in an image',
      'To separate the image into regions with different attributes',
      'To enhance image resolution',
      'To apply color transformations',
    ],
    answer: 'To separate the image into regions with different attributes',
    explanation:
      'Image segmentation is used to divide an image into distinct regions or objects, which simplifies analysis and helps in identifying specific areas of interest.',
  },
  {
    id: 215,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question: 'What is global thresholding in image segmentation?',
    options: [
      'A technique that applies a single threshold value to the entire image',
      'A technique that adjusts the threshold value for each pixel individually',
      'A method for reducing noise in an image',
      'A method for detecting edges',
    ],
    answer:
      'A technique that applies a single threshold value to the entire image',
    explanation:
      'Global thresholding involves using one global threshold value for the entire image to convert it into a binary format, distinguishing foreground and background.',
  },
  {
    id: 216,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question: 'What is adaptive thresholding in image segmentation?',
    options: [
      'A method that applies a single threshold value across the image',
      'A technique that adjusts the threshold value based on local pixel intensity',
      'A method that detects edges in the image',
      'A technique that enhances contrast in low-light images',
    ],
    answer:
      'A technique that adjusts the threshold value based on local pixel intensity',
    explanation:
      'Adaptive thresholding adjusts the threshold for different regions of an image, making it suitable for images with varying lighting conditions.',
  },
  {
    id: 217,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question: 'What is contour detection in image processing?',
    options: [
      'The process of identifying and outlining objects in an image',
      'A technique for reducing image noise',
      'The process of detecting edges in an image',
      'A method for color enhancement',
    ],
    answer: 'The process of identifying and outlining objects in an image',
    explanation:
      'Contour detection is used to identify the boundaries of objects in an image, making it easier to separate and analyze individual objects.',
  },
  {
    id: 218,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question: 'What does the SURF (Speeded-Up Robust Features) detector do?',
    options: [
      'It detects corners and edges in an image',
      'It detects and extracts features that are invariant to scaling and rotation',
      'It enhances image resolution',
      'It filters out noise from an image',
    ],
    answer:
      'It detects and extracts features that are invariant to scaling and rotation',
    explanation:
      'SURF is a feature detection algorithm that identifies key points in an image and is robust to transformations like scaling and rotation.',
  },
  {
    id: 219,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question:
      'What does the SIFT (Scale-Invariant Feature Transform) detector do?',
    options: [
      'It detects key points that are invariant to scale and rotation',
      'It enhances the edges of an image',
      'It converts an image to grayscale',
      'It reduces image noise',
    ],
    answer: 'It detects key points that are invariant to scale and rotation',
    explanation:
      'SIFT is an algorithm that detects key points and features in images that remain invariant under transformations such as scaling, rotation, and affine transformations.',
  },
  {
    id: 220,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question:
      'What is the purpose of the ORB (Oriented FAST and Rotated BRIEF) detector?',
    options: [
      'To detect key points and describe them using binary features',
      'To enhance edges in an image',
      'To classify objects in an image',
      'To filter out background noise',
    ],
    answer: 'To detect key points and describe them using binary features',
    explanation:
      'ORB is a feature detector and descriptor that is faster and more efficient than SIFT and SURF. It uses binary features for key point matching and is rotation invariant.',
  },
  {
    id: 221,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question: 'What is the process of feature matching in image processing?',
    options: [
      'Matching pixel values between two images',
      'Comparing key points between two images to identify similarities',
      'Reducing image resolution for faster processing',
      'Removing noise from an image',
    ],
    answer: 'Comparing key points between two images to identify similarities',
    explanation:
      'Feature matching involves comparing key points detected in two images to identify similarities, which is useful in tasks like object recognition and image stitching.',
  },
  {
    id: 222,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question:
      'Which of the following is a common use of segmentation in machine vision?',
    options: [
      'Face detection',
      'Object detection and recognition',
      'Image compression',
      'Image enhancement',
    ],
    answer: 'Object detection and recognition',
    explanation:
      'Segmentation is commonly used to identify and separate objects in an image, which is a crucial step for tasks like object detection and recognition.',
  },
  {
    id: 223,
    topic: [TopicEnum.MV, TopicEnum.SegmentationOD],
    question:
      'Which technique is often used to detect the boundary of objects in an image?',
    options: [
      'Canny edge detection',
      'SIFT detection',
      'Gaussian blur',
      'Data augmentation',
    ],
    answer: 'Canny edge detection',
    explanation:
      'Canny edge detection is widely used to detect edges and boundaries in images by identifying areas where there is a rapid change in pixel intensity.',
  },
  {
    id: 224,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question:
      'What is one of the main differences between traditional methods and neural networks for image classification?',
    options: [
      'Traditional methods require manual feature extraction, while neural networks learn features automatically',
      'Neural networks require manual feature extraction, while traditional methods do not',
      'Neural networks are faster than traditional methods',
      'Traditional methods use deep learning algorithms',
    ],
    answer:
      'Traditional methods require manual feature extraction, while neural networks learn features automatically',
    explanation:
      'In traditional methods, features need to be manually extracted and defined, whereas neural networks automatically learn the relevant features during the training process.',
  },
  {
    id: 225,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question:
      'Which of the following is a limitation of Artificial Neural Networks (ANN) for image classification?',
    options: [
      'Requires large amounts of labeled data for training',
      'Faster than traditional methods',
      'Does not require significant computational resources',
      'Only works on grayscale images',
    ],
    answer: 'Requires large amounts of labeled data for training',
    explanation:
      'ANNs require large datasets to train effectively, as they need sufficient data to learn the underlying patterns and features in the images.',
  },
  {
    id: 226,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question:
      'Which of the following is NOT a component of Convolutional Neural Networks (CNNs)?',
    options: [
      'Convolutional layers',
      'ReLU activation layers',
      'Pooling layers',
      'Histogram equalization',
    ],
    answer: 'Histogram equalization',
    explanation:
      'Histogram equalization is a preprocessing technique for improving image contrast, not a component of CNNs. CNNs consist of convolutional layers, pooling layers, and ReLU activation layers.',
  },
  {
    id: 227,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question: 'What is the role of a convolutional layer in a CNN?',
    options: [
      'It extracts spatial features from the image',
      'It reduces the image size',
      'It applies non-linear activation functions',
      'It combines features from different layers',
    ],
    answer: 'It extracts spatial features from the image',
    explanation:
      'The convolutional layer in a CNN is responsible for detecting spatial hierarchies by applying filters to extract features such as edges, textures, and patterns.',
  },
  {
    id: 228,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question: 'What is the purpose of pooling layers in a CNN?',
    options: [
      'To reduce the dimensionality of feature maps',
      'To apply activation functions',
      'To extract spatial features',
      'To generate new training data',
    ],
    answer: 'To reduce the dimensionality of feature maps',
    explanation:
      'Pooling layers help reduce the spatial dimensions of feature maps, which reduces the number of parameters and computation required, while retaining important features.',
  },
  {
    id: 229,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question:
      'Which of the following metrics is used to evaluate the performance of an image classification model?',
    options: [
      'ROC curve',
      'Mean squared error',
      'Precision',
      'All of the above',
    ],
    answer: 'ROC curve',
    explanation:
      'The ROC curve is commonly used to evaluate the performance of classification models, particularly for binary classification tasks, by plotting the true positive rate versus the false positive rate.',
  },
  {
    id: 230,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question:
      'What is one of the main advantages of using neural networks for image classification compared to traditional methods?',
    options: [
      'Neural networks can automatically learn complex features from raw data',
      'Neural networks require no data preprocessing',
      'Neural networks do not require large datasets',
      'Traditional methods are better at handling complex patterns',
    ],
    answer:
      'Neural networks can automatically learn complex features from raw data',
    explanation:
      'Neural networks can automatically learn hierarchical features from raw image data without the need for manual feature extraction, which makes them more powerful for complex tasks.',
  },
  {
    id: 231,
    topic: [TopicEnum.MV, TopicEnum.CNN],
    question:
      'What is a common challenge with using artificial neural networks (ANN) for image classification?',
    options: [
      'Overfitting due to the large number of parameters',
      'Difficulty in extracting features',
      'Lack of interpretability',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      'ANNs can suffer from overfitting due to the large number of parameters, they may require significant amounts of labeled data, and their black-box nature makes them difficult to interpret.',
  },
  {
    id: 232,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question: 'What is the purpose of data augmentation in deep learning?',
    options: [
      'To artificially increase the size of the training dataset by generating modified versions of the images',
      'To reduce the complexity of the model',
      'To decrease the training time',
      'To remove noise from the data',
    ],
    answer:
      'To artificially increase the size of the training dataset by generating modified versions of the images',
    explanation:
      'Data augmentation generates variations of the training data, such as rotations or flips, to increase the diversity of the dataset and prevent overfitting.',
  },
  {
    id: 233,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question: 'Which of the following is a common method of data augmentation?',
    options: [
      'Image rotation',
      'Image flipping',
      'Zooming in and out',
      'All of the above',
    ],
    answer: 'All of the above',
    explanation:
      "Common data augmentation methods include rotating, flipping, and zooming in and out of images, which help increase the model's ability to generalize.",
  },
  {
    id: 234,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question:
      'What is one of the key considerations when designing a CNN architecture?',
    options: [
      'Choosing an appropriate number of layers and neurons to avoid overfitting',
      'Choosing the size of the images in the training dataset',
      'Ensuring the dataset contains only grayscale images',
      'Using a small number of convolutional layers',
    ],
    answer:
      'Choosing an appropriate number of layers and neurons to avoid overfitting',
    explanation:
      'When designing CNN architectures, it is crucial to balance the number of layers and neurons to prevent overfitting while ensuring sufficient model capacity.',
  },
  {
    id: 235,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question:
      'Which of the following CNN components helps the network learn hierarchical features?',
    options: [
      'Convolutional layers',
      'Fully connected layers',
      'Recurrent layers',
      'Normalization layers',
    ],
    answer: 'Convolutional layers',
    explanation:
      'Convolutional layers in CNNs are responsible for learning hierarchical spatial features from raw image data by applying filters to the input image.',
  },
  {
    id: 236,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question:
      'Which technique is commonly used in CNNs to prevent overfitting?',
    options: ['Dropout', 'Image resizing', 'Normalization', 'Backpropagation'],
    answer: 'Dropout',
    explanation:
      'Dropout is a regularization technique used in CNNs to randomly deactivate neurons during training to prevent overfitting and improve generalization.',
  },
  {
    id: 237,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question: 'What is the benefit of using batch normalization in CNNs?',
    options: [
      'To speed up training by normalizing input layers',
      'To reduce computational cost',
      'To increase the depth of the network without overfitting',
      'To create more complex models',
    ],
    answer: 'To speed up training by normalizing input layers',
    explanation:
      'Batch normalization normalizes the input to each layer, which helps speed up the training process and stabilize the learning process by reducing internal covariate shift.',
  },
  {
    id: 238,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question: 'What is the advantage of using pooling layers in a CNN?',
    options: [
      'To reduce the spatial dimensions of the feature maps',
      'To apply activation functions',
      'To learn features from the data',
      'To prevent overfitting',
    ],
    answer: 'To reduce the spatial dimensions of the feature maps',
    explanation:
      'Pooling layers reduce the dimensionality of feature maps, helping to decrease computational load and the number of parameters in the model.',
  },
  {
    id: 239,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question:
      'How does the use of data augmentation help with model generalization?',
    options: [
      'By increasing the variety of training data, making the model less likely to overfit',
      'By making the model train faster',
      'By reducing the need for hyperparameter tuning',
      'By decreasing the complexity of the model',
    ],
    answer:
      'By increasing the variety of training data, making the model less likely to overfit',
    explanation:
      'Data augmentation increases the variety of training data by introducing transformations, thus helping the model generalize better to unseen data.',
  },
  {
    id: 240,
    topic: [TopicEnum.MV, TopicEnum.DLCNN],
    question:
      'Which of the following is not a common method for adjusting the architecture of a CNN during hyperparameter tuning?',
    options: [
      'Changing the number of filters in convolutional layers',
      'Adjusting the size of the training dataset',
      'Changing the type of activation function',
    ],
    answer: 'Adjusting the size of the training dataset',
    explanation:
      'During hyperparameter tuning, one may adjust the number of filters in convolutional layers and experiment with different activation functions to improve model performance.',
  },
  {
    id: 241,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which of the following is a major difference between PyTorch and TensorFlow?',
    options: [
      'PyTorch uses dynamic computation graphs, while TensorFlow uses static computation graphs',
      'PyTorch is designed for mobile applications, while TensorFlow is for desktop applications',
      'TensorFlow is primarily used for computer vision, while PyTorch is used for NLP',
      'PyTorch requires more memory than TensorFlow',
    ],
    answer:
      'PyTorch uses dynamic computation graphs, while TensorFlow uses static computation graphs',
    explanation:
      'PyTorch uses dynamic computation graphs (define-by-run), while TensorFlow uses static computation graphs (define-and-run), giving PyTorch more flexibility during model development.',
  },
  {
    id: 242,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'What is the primary data structure used in PyTorch for storing multi-dimensional data?',
    options: ['Tensor', 'DataFrame', 'Matrix', 'Array'],
    answer: 'Tensor',
    explanation:
      'In PyTorch, a Tensor is the primary data structure used for storing multi-dimensional arrays, and it is similar to NumPy arrays but can be run on GPUs for faster computation.',
  },
  {
    id: 243,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      "Which of the following is a key component of PyTorch's neural network module?",
    options: ['torch.nn', 'torch.optim', 'torch.autograd', 'All of the above'],
    answer: 'All of the above',
    explanation:
      "PyTorch's neural network module includes several components like torch.nn (for defining models), torch.optim (for optimization algorithms), and torch.autograd (for automatic differentiation).",
  },
  {
    id: 244,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which of the following is the most appropriate use case for TensorFlow?',
    options: [
      'Real-time inference and deployment of models in production',
      'Research and rapid prototyping',
      'Training models on mobile devices',
      'Manipulating tensors for low-level tasks',
    ],
    answer: 'Real-time inference and deployment of models in production',
    explanation:
      'TensorFlow is known for its strengths in deploying models to production, especially in environments requiring real-time inference and scalable systems.',
  },
  {
    id: 245,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: "What is the role of the 'autograd' feature in PyTorch?",
    options: [
      'To automatically compute gradients for backpropagation',
      'To optimize memory usage during training',
      'To manage data input pipelines',
      'To provide debugging capabilities',
    ],
    answer: 'To automatically compute gradients for backpropagation',
    explanation:
      "'autograd' in PyTorch handles automatic differentiation, allowing for efficient computation of gradients during backpropagation, essential for training neural networks.",
  },
  {
    id: 246,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      "Which of the following statements is true about PyTorch's dynamic computation graph?",
    options: [
      'It allows for flexible modification of the model during runtime',
      "It is faster than TensorFlow's static computation graph",
      'It requires less memory than TensorFlow',
      'It is difficult to implement for large models',
    ],
    answer: 'It allows for flexible modification of the model during runtime',
    explanation:
      "PyTorch's dynamic computation graph enables flexibility as the model can be modified during runtime, making debugging and prototyping easier compared to static graphs in TensorFlow.",
  },
  {
    id: 247,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which of the following is a core benefit of using PyTorch for model development?',
    options: [
      'Ease of debugging due to dynamic computation graphs',
      'Better support for mobile applications',
      'Faster training due to pre-built optimizations',
      'Lower memory consumption compared to TensorFlow',
    ],
    answer: 'Ease of debugging due to dynamic computation graphs',
    explanation:
      "PyTorch's dynamic computation graph makes it easier to debug models as the computation graph is built on-the-fly, which allows for immediate feedback during model development.",
  },
  {
    id: 248,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question:
      'Which of the following is an important consideration when using PyTorch for large-scale distributed training?',
    options: [
      "Ensuring that the model is compatible with PyTorch's data parallelism tools",
      "Using TensorFlow's data parallelism techniques",
      'Minimizing the size of the model to fit into memory',
      'Limiting the training time',
    ],
    answer:
      "Ensuring that the model is compatible with PyTorch's data parallelism tools",
    explanation:
      "For large-scale distributed training in PyTorch, it is important to use tools like 'torch.nn.DataParallel' or 'torch.distributed' to ensure efficient parallelism across multiple devices.",
  },
  {
    id: 249,
    topic: [TopicEnum.MV, TopicEnum.PyTorch],
    question: "What is the purpose of the 'torch.nn.Module' class in PyTorch?",
    options: [
      'To define and organize the layers and operations of a neural network',
      'To handle data loading and preprocessing',
      'To define the optimization algorithm used during training',
      'To automatically update model weights during training',
    ],
    answer:
      'To define and organize the layers and operations of a neural network',
    explanation:
      "'torch.nn.Module' is the base class for all neural network modules in PyTorch, and it allows users to define the structure and behavior of the neural network.",
  },
];
