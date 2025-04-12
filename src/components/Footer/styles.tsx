import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center; /* Centers the content horizontally */
  padding: 20px;
  justify-content: flex-end; /* Ensures the footer stays at the bottom */

  @media (max-width: 768px) {
    flex-direction: column; /* Stack content vertically on smaller screens */
    padding: 15px; /* Adjust padding for mobile */
  }
`;

export const GreenText = styled.span`
  color: ${(props) => props.theme.colors.darkGreen};
`;

export const CopyrightText = styled.p`
  font-size: 0.9rem;
  margin-top: 10px;
  text-align: center;
  margin-bottom: 10px; /* Ensures space at the bottom for mobile view */
`;
