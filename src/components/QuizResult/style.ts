import styled from 'styled-components';

export const ResultWrapper = styled.div`
  padding: 20px;
  background-color: #f4f4f9;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

export const ScoreText = styled.h2`
  text-align: center;
  color: #333;
  font-size: 24px;
  margin-bottom: 20px;
`;

export const IncorrectAnswersWrapper = styled.div`
  margin-top: 20px;
  background-color: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

export const IncorrectAnswerCard = styled.div`
  padding: 10px;
  margin-bottom: 15px;
  border-left: 4px solid ${(props) => props.theme.colors.lightRed};
  background-color: ${(props) => props.theme.colors.lightGray};
`;

export const QuestionText = styled.p`
  font-weight: bold;
  margin-bottom: 5px;
  font-size: 16px;
`;

export const AnswerText = styled.p`
  margin-bottom: 5px;
  font-size: 14px;
`;

export const StyledExplanationText = styled.p`
  font-size: 14px;
  color: #666;
`;
