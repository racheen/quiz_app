import styled from 'styled-components';

export const Card = styled.div`
  padding: 2rem;
  border-radius: 1rem;
  background-color: ${(props) => props.theme.colors.secondary};
  box-shadow: 0 4px 10px ${(props) => props.theme.shadows.card};
`;

export const QuestionText = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color:  ${(props) => props.theme.colors.text};
`;

export const OptionButton = styled.button`
  display: block;
  width: 100%;
  text-align: left;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  background-color: ${(props) => props.theme.colors.accent};
  border: 2px solid ${(props) => props.theme.colors.lightGray};
  border-radius: 0.5rem;
  font-size: ${(props) => props.theme.fontSizes.normal};
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: ${(props) => props.theme.colors.darkGray};
  }
`;
