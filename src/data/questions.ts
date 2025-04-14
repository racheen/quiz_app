import { Question } from '../types/question';

export enum TopicEnum {
  DataPreprocessing = 'Data Preprocessing',
  SupportVectorMachines = 'Support Vector Machines',
  NeuralNetworks = 'Neural Networks',
  TimeSeriesRNN = 'Time Series RNN',
  NaiveBayes = 'Naive Bayes',
  Clustering = 'Clustering',
  HyperparameterTuning = 'Hyperparameter Tuning',
  Visualizations = 'Visualizations',
  AML = 'Advanced Machine Leraning',
  ClassifierFusion = 'Classifier Fusion',
  ScikitLearn = 'Scikit Learn',
}

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
      'LDA',
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
    question:
      'OCSVM developed by Tax & Duin uses the concept of',
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
    question:
      'OCSVM creates a hyperplane between',
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
    question:
      'Which method uses weak stable classifiers?',
    options: [
      'Bagging',
      'Majority Vote',
      'All of these',
      'Boosting',
    ],
    answer: 'Boosting',
    explanation:
      'Boosting uses weak learners (often decision stumps) and focuses on improving them sequentially to form a strong ensemble classifier.',
  },
  {
    id: 118,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'Bagging is',
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
    options: [
      'Boosting',
      'Majority vote',
      'Bagging',
      'All of these',
    ],
    answer: 'Bagging',
    explanation:
      'Bagging helps reduce variance by training each model on different subsets of data and averaging the results, thereby reducing overfitting.',
  },
  {
    id: 120,
    topic: [TopicEnum.AML, TopicEnum.ClassifierFusion],
    question:
      'Random Forest uses',
    options: [
      'Bagging',
      'Boosting',
      'Both of these',
      'Any of these',
    ],
    answer: 'Bagging',
    explanation:
      'Random Forest is an ensemble learning method that builds multiple decision trees using the Bagging technique.',
  },
  {
    id: 121,
    topic: [TopicEnum.AML, TopicEnum.Clustering],
    question:
      'OCSVM that creates a hyperplane',
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
    question:
      'The harmonic mean of precision and recall is',
    options: [
      'F1 score',
      'Accuracy',
      'Specificity',
      'Focal Loss',
    ],
    answer: 'F1 score',
    explanation:
      'F1 score is the harmonic mean of precision and recall, providing a balance between the two in evaluating classification models.',
  },
  {
    id: 123,
    topic: [TopicEnum.AML],
    question:
      'SMOTE is',
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
    question:
      'OCSVM can be used for',
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
];
