import styled from 'styled-components';

export const SelectorWrapper = styled.div`
  padding: 2rem;
  border-radius: 1rem;
  background-color: ${(props) => props.theme.colors.background};
  text-align: center;
  box-shadow: ${(props) => props.theme.shadows.card};
`;

export const TopicButton = styled.button`
  padding: 0.75rem 2rem;
  margin: 1rem;
  background-color: ${(props) => props.theme.colors.lightGray};
  color: ${(props) => props.theme.colors.text};
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.3s;

  &:hover {
    background-color: ${(props) => props.theme.colors.darkGray};
  }
`;
