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
      {question.options.map((option, index) => {
        const isSelected = selectedAnswer === option;
        const isCorrect = selectedAnswer !== null && option === question.answer;
        const isIncorrect = isSelected && option !== question.answer;

        return (
          <OptionButton
            key={index}
            onClick={() => onAnswer(option)}
            disabled={selectedAnswer !== null}
            isCorrect={Boolean(isCorrect)}
            isIncorrect={Boolean(isIncorrect)}
          >
            <DynamicText text={option} />
          </OptionButton>
        );
      })}
      {selectedAnswer && selectedAnswer !== question.answer && (
        <div style={{ marginTop: '1rem', color: '#e63946' }}>
          <strong>Explanation:</strong>{' '}
          <DynamicText text={question.explanation} />
        </div>
      )}
    </Card>
  );
};

export default QuestionCard;
