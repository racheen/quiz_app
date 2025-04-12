import React from 'react';
import {
  AnswerText,
  IncorrectAnswerCard,
  IncorrectAnswersWrapper,
  QuestionText,
  ResultWrapper,
  ScoreText,
} from './style';
import 'katex/dist/katex.min.css';

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
                <strong>Question:</strong> {answer.question}
              </QuestionText>
              <AnswerText>
                <strong>Your Answer:</strong> {answer.chosenAnswer}
              </AnswerText>
              <AnswerText>
                <strong>Correct Answer:</strong> {answer.correctAnswer}
              </AnswerText>
            </IncorrectAnswerCard>
          ))}
        </IncorrectAnswersWrapper>
      )}
    </ResultWrapper>
  );
};

export default QuizResult;
