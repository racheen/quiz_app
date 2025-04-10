import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start; /* Left-aligns the content */
  padding: 20px;
  min-height: 100vh; /* Ensures the container takes full height */
  justify-content: flex-end; /* Pushes content to the bottom */
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  width: 100%;

  @media (max-width: 768px) {
    /* Ensure it is positioned at the bottom when in mobile view */
    position: relative;
    min-height: auto; /* Ensures it takes full screen height */
    justify-content: flex-end; /* Push content to the bottom */
  }
`;

export const GreenText = styled.span`
  color: ${(props) => props.theme.colors.darkGreen};
`;

export const CopyrightText = styled.p`
  font-size: 0.9rem;
  margin-top: 10px;
  text-align: center;
`;

export const SocialIcons = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 15px;
`;
