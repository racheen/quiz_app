import styled from 'styled-components';

export const Card = styled.div`
  padding: 2rem;
  border-radius: 1rem;
  background-color: ${(props) => props.theme.colors.background};
  box-shadow: ${(props) => props.theme.shadows.card};
`;

export const QuestionText = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: ${(props) => props.theme.colors.text};
`;

export const OptionButton = styled.button<{ selected?: boolean }>`
  display: block;
  width: 100%;
  text-align: left;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  background-color: ${({ selected, theme }) =>
    selected ? theme.colors.darkGray : theme.colors.lightGray};
  border: 2px solid
    ${({ selected, theme }) =>
      selected ? theme.colors.lightGray : theme.colors.accent};
  color: ${({ selected, theme }) => (selected ? theme.colors.text : 'inherit')};
  border-radius: 0.5rem;
  font-size: ${({ theme }) => theme.fontSizes.normal};
  cursor: pointer;
  transition: background-color 0.2s, border 0.2s;

  &:hover {
    background-color: ${({ selected, theme }) =>
      selected ? theme.colors.accent : theme.colors.darkGray};
  }

  &:disabled {
    opacity: 0.7;
    cursor: default;
  }
`;
