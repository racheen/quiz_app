import React from 'react';
import { Question } from '../../types/question';
import { Card, OptionButton, QuestionText } from './style';

type Props = {
  question: Question;
  onAnswer: (answer: string) => void;
};

const QuestionCard: React.FC<Props> = ({ question, onAnswer }) => {
  return (
    <Card>
      <QuestionText>{question.question}</QuestionText>
      {question.options.map((option, index) => (
        <OptionButton key={index} onClick={() => onAnswer(option)}>
          {option}
        </OptionButton>
      ))}
    </Card>
  );
};

export default QuestionCard;
