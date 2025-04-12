import React from 'react';
import { SelectorWrapper, TopicButton } from './style';

type Props = {
  topics: string[];
  onSelectTopic: (topic: string) => void;
};

const TopicSelector: React.FC<Props> = ({ topics, onSelectTopic }) => {
  const handleClick = (topic: string) => {
    console.log('Topic selected:', topic); // Debug log
    onSelectTopic(topic); // Make sure this function works as expected
  };

  return (
    <SelectorWrapper>
      <h2>Select a Topic</h2>
      {topics.map((topic, index) => (
        <>
          <TopicButton key={index} onClick={() => handleClick(topic)}>
            {topic}
          </TopicButton>
        </>
      ))}
    </SelectorWrapper>
  );
};

export default TopicSelector;
