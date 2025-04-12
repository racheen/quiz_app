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
      'Principal Component Analysis (PCis a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It does not require labels, meaning it does not rely on supervision, making it an unsupervised learning algorithm.',
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
      "Logistic Regression is a classification algorithm. It is used to model the probability of a binary outcome (1 or 0) and makes predictions based on a logistic (sigmoifunction. It's widely used for binary classification tasks.",
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
  {
    id: 15,
    topic: 'AML',
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
    topic: 'AML',
    question: 'Same padding means',
    options: [
      'add 1 padding on both sides',
      'No padding',
      'add padding such that the convoluted output matrix size should be the same as the input matrix size',
      'if the image size is nxn, add same n pixels on both sides',
    ],
    answer: 'Position and orientation of the hyperplane',
    explanation:
      'Same padding means that padding is added to the input image in such a way that the output size of the convolution is the same as the input size. The padding ensures that the filter can slide across the entire image without reducing the spatial dimensions of the input.',
  },
  {
    id: 17,
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
    question: 'The hyperplane of the SVM of a dataset with 2 features is a',
    options: ['line', 'point', 'circle', 'plane'],
    answer: 'line',
    explanation:
      'In a dataset with 2 features, the data lies in a 2D plane. In this case, the SVM hyperplane is a line that separates the classes.',
  },
  {
    id: 22,
    topic: 'AML',
    question:
      'If noise dominates in your dataset, which technique is the best to reduce the dimensionality of this dataset?',
    options: [
      'both are equivalent',
      'PCA',
      'Cannot tell, it depends on the attributes',
      'LDA',
    ],
    answer: 'PCA',
    explanation:
      'PCA is an unsupervised technique that focuses on directions of maximum variance. When noise dominates the dataset, PCA helps by reducing dimensions and removing components associated with low variance (often noise), improving signal clarity.',
  },
  {
    id: 23,
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
    question: 'The conditional probability P(A|is',
    options: [
      '$P(A|= \\frac{P(A∪B)}{P(B)}$',
      '$P(A|= \\frac{P(A∩B)}{P(B)}$',
      '$P(A|= \\frac{P(A∩B)}{P(A)}$',
      '$P(A|= \\frac{P(A∪B)}{P(A)}$',
    ],
    answer: '$P(A|= \\frac{P(A∩B)}{P(B)}$',
    explanation:
      'Conditional probability is defined as the probability of event A given that event B has occurred. It is calculated using the formula: \n\n$$P(A|= \\frac{P(A∩B)}{P(B)}$$',
  },
  {
    id: 26,
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
    question:
      'Which type of join can Tableau use when combining data from multiple tables?',
    options: ['Left Join', 'Inner Join', 'Outer Join', 'All of the above'],
    answer: 'All of the above',
    explanation:
      'Tableau supports multiple types of joins when combining data from multiple tables, including left join, inner join, and outer join, depending on the analysis requirements.',
  },
  {
    id: 41,
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
    topic: 'AML',
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
];
