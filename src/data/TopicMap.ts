// utils/topicMap.ts
import { MainTopic, TopicEnum } from './questions';

export const topicHierarchy: Record<MainTopic, TopicEnum[]> = {
  [MainTopic.MV]: [
    TopicEnum.MVIntro,
    TopicEnum.ImageProcessing,
    TopicEnum.SegmentationOD,
    TopicEnum.CNN,
    TopicEnum.DLCNN,
    TopicEnum.ObjectDetection,
    TopicEnum.ObjectTracking,
    TopicEnum.SensorFusion,
  ],
  [MainTopic.AML]: [
    TopicEnum.DataPreprocessing,
    TopicEnum.SupportVectorMachines,
    TopicEnum.NeuralNetworks,
    TopicEnum.TimeSeriesRNN,
    TopicEnum.NaiveBayes,
    TopicEnum.Clustering,
    TopicEnum.HyperparameterTuning,
    TopicEnum.Visualizations,
    TopicEnum.ClassifierFusion,
    TopicEnum.ScikitLearn,
    TopicEnum.PyTorch,
  ],
  [MainTopic.RL]: [],
};
