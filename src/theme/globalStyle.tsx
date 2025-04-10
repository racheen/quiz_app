import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`

  @font-face {
    font-family: "Exo2";
    src: local("Exo2"),
      url(./assets/fonts/Exo2-VariableFont_wght.ttf) format("truetype");
  }

  body {
    font-family: 'Exo2', sans-serif;
    background-color: ${({ theme }) => theme.colors.background};
    color: ${({ theme }) => theme.colors.text};
  }
`;

export default GlobalStyle;
