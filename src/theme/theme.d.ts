import 'styled-components';

// Define the structure of your theme
declare module 'styled-components' {
  export interface DefaultTheme {
    colors: {
      white: string;
      lightGray: string;
      lightGreen: string;
      darkGreen: string;
      darkGray: string;
      mediumGray: string;
      lightRed: string;
      primary: string;
      secondary: string;
      text: string;
      accent: string;
      background: string;
    };
    fontWeights: {
      regular: number;
      medium: number;
      semiBold: number;
      bold: number;
      light: number;
    };
    fontSizes: {
      xLarge: string;
      large: string;
      medium: string;
      normal: string;
      small: string;
      h1: string;
      h2: string;
      h3: string;
      h4: string;
    };
    size: {
      desktop: string;
      laptop: string;
      laptopL: string;
      tablet: string;
      mobileL: string;
      mobileM: string;
      mobileS: string;
    };
    device: {
      desktop: string;
      desktopL: string;
      laptop: string;
      tablet: string;
      mobileL: string;
      mobileM: string;
      mobileS: string;
    };
    shadows: {
      dropShadowCard: string;
      card: string;
      hover: string;
    };
    gradients: {
      darkGreenLightGreen: string;
      lightGreenWhite: string;
      darkGreenWhite: string;
      mediumGrayLightGreen: string;
      darkGrayWhite: string;
      mediumGrayLightRed: string;
    };
    id: number;
  }
}
