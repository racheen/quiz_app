import React from 'react';
import { ResultWrapper, ScoreText } from './style';

type Props = {
  score: number;
  total: number;
};

const QuizResult: React.FC<Props> = ({ score, total }) => {
  return (
    <ResultWrapper>
      <ScoreText>
        You scored {score} out of {total}!
      </ScoreText>
    </ResultWrapper>
  );
};

export default QuizResult;
