import React from 'react';
import {
  AnswerText,
  IncorrectAnswerCard,
  IncorrectAnswersWrapper,
  QuestionText,
  ResultWrapper,
  ScoreText,
  StyledExplanationText,
} from './style';
import 'katex/dist/katex.min.css';
import DynamicText from '../DynamicText';

type Answer = {
  question: string;
  chosenAnswer: string;
  correctAnswer: string;
  explanation: string;
};

type Props = {
  score: number;
  total: number;
  incorrectAnswers: Answer[];
};

const QuizResult: React.FC<Props> = ({ score, total, incorrectAnswers }) => {
  return (
    <ResultWrapper>
      <ScoreText>
        You scored {score} out of {total}!
      </ScoreText>

      {incorrectAnswers.length > 0 && (
        <IncorrectAnswersWrapper>
          <h3>Incorrect Answers</h3>
          {incorrectAnswers.map((answer, index) => (
            <IncorrectAnswerCard key={index}>
              <QuestionText>
                <strong>Question:</strong>{' '}
                <DynamicText text={answer.question} />
              </QuestionText>
              <AnswerText>
                <strong>Your Answer:</strong>{' '}
                <DynamicText text={answer.chosenAnswer} />
              </AnswerText>
              <AnswerText>
                <strong>Correct Answer:</strong>{' '}
                <DynamicText text={answer.correctAnswer} />
              </AnswerText>
              <StyledExplanationText>
                <DynamicText text={answer.explanation} />
              </StyledExplanationText>
            </IncorrectAnswerCard>
          ))}
        </IncorrectAnswersWrapper>
      )}
    </ResultWrapper>
  );
};

export default QuizResult;
