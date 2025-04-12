import styled from 'styled-components';

export const ButtonContainer = styled.div`
  display: flex;
  justify-content: center; /* Centers horizontally */
  align-items: center; /* Centers vertically */
  padding: 20px;
`;

export const BackToHomeButton = styled.button`
  background-color: ${(props) =>
    props.theme.colors.primary}; /* Primary color */
  color: ${(props) => props.theme.colors.text};
  font-size: 1rem;
  padding: 12px 24px;
  border-radius: 30px; /* Rounded corners */
  border: none;
  cursor: pointer;
  text-align: center;
  transition: background-color 0.3s ease, transform 0.3s ease;
  box-shadow: ${(props) => props.theme.shadows.card};

  &:hover {
    background-color: ${(props) =>
      props.theme.colors.lightGray}; /* Darker shade for hover */
    transform: scale(1.05); /* Slight zoom-in effect on hover */
  }

  &:active {
    background-color: ${(props) =>
      props.theme.colors.darkGray}; /* Lighter shade for active state */
    transform: scale(0.98); /* Slight shrink effect on click */
  }

  &:focus {
    outline: none;
    box-shadow: ${(props) =>
      props.theme.shadows.hover}; /* Subtle shadow for focus */
  }

  @media (max-width: 768px) {
    font-size: 0.9rem; /* Slightly smaller font size on mobile */
    padding: 10px 20px;
  }
`;
