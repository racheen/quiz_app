import React from 'react';
import 'katex/dist/katex.min.css';

import { Question } from '../../types/question';
import { Card, OptionButton, QuestionText } from './style';
import DynamicText from '../DynamicText';

type Props = {
  question: Question;
  onAnswer: (answer: string) => void;
  selectedAnswer: string | null;
};

const QuestionCard: React.FC<Props> = ({
  question,
  onAnswer,
  selectedAnswer,
}) => {
  return (
    <Card>
      <QuestionText>
        <DynamicText text={question.question} />
      </QuestionText>
      {question.options.map((option, index) => (
        <OptionButton
          key={index}
          onClick={() => onAnswer(option)}
          selected={selectedAnswer === option}
        >
          <DynamicText text={option} />
        </OptionButton>
      ))}
    </Card>
  );
};

export default QuestionCard;
