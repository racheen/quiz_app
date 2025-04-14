import styled from 'styled-components';

export const Card = styled.div`
  padding: 2rem;
  border-radius: 1rem;
  background-color: ${(props) => props.theme.colors.background};
  box-shadow: ${(props) => props.theme.shadows.card};
  width: 300px;
`;

export const QuestionText = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: ${(props) => props.theme.colors.text};
`;

export const OptionButton = styled.button<{
  selected?: boolean | undefined;
  isCorrect?: boolean | undefined;
  isIncorrect?: boolean | undefined;
}>`
  display: block;
  width: 100%;
  text-align: left;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  border-radius: 0.5rem;
  font-size: ${({ theme }) => theme.fontSizes.normal};
  cursor: pointer;
  transition: background-color 0.2s, border 0.2s;

  background-color: ${({ theme, selected, isCorrect, isIncorrect }) =>
    isCorrect
      ? 'lightgreen'
      : isIncorrect
      ? 'lightcoral'
      : theme.colors.lightGray};

  border: 2px solid
    ${({ theme, selected }) =>
      selected ? theme.colors.lightGray : theme.colors.accent};

  color: ${({ selected, theme }) => (selected ? theme.colors.text : 'inherit')};

  &:hover {
    background-color: ${({ theme, selected, isCorrect, isIncorrect }) =>
      isCorrect
        ? 'lightgreen'
        : isIncorrect
        ? 'lightcoral'
        : theme.colors.lightGray};
  }

  &:disabled {
    opacity: 0.7;
    cursor: default;
  }
`;
