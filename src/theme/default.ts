/* eslint-disable no-magic-numbers */
import { rem } from 'polished';

const id = 1;

const colors = {
  white: '#FCFCFC',
  lightGray: '#CCCCCC',
  lightGreen: '#BBDED7',
  darkGreen: '#80AB82',
  darkGray: '#7F7F7F',
  mediumGray: '#595959',
  lightRed: '#DF8C87',
  primary: '#BBDED7',     // Light Green (Primary)
  secondary: '#7DCD85',   // Dark Green (Secondary)
  text: '#595959',        // Dark Gray (Text Color)
  accent: '#C2E1C2',      // Event Card Background Color
  background: '#FCFCFC',  // Default Background
};

const fontWeights = {
  regular: 400,
  medium: 500,
  semiBold: 600,
  bold: 700,
  light: 300,
};

const fontSizes = {
  xLarge: '36px',
  large: rem('24px'),
  medium: rem('20px'),
  normal: rem('16px'),
  small: rem('12px'),
  h1: rem('80px'),
  h2: rem('60px'),
  h3: rem('30px'),
  h4: rem('24px'),
};

const size = {
  desktop: '2560px',
  laptop: '1024px',
  laptopL: '1440px',
  tablet: '768px',
  mobileL: '426px',
  mobileM: '376px',
  mobileS: '320px',
};

const device = {
  desktop: `(max-width: ${size.desktop})`,
  desktopL: `(max-width: ${size.laptopL})`,
  laptop: `(max-width: ${size.laptop})`,
  tablet: `(max-width: ${size.tablet})`,
  mobileL: `(max-width: ${size.mobileL})`,
  mobileM: `(max-width: ${size.mobileM})`,
  mobileS: `(max-width: ${size.mobileS})`,
};

const shadows = {
  dropShadowCard: '1px 2px 4px #BBDED7',
  card: '0px 4px 6px rgba(0, 0, 0, 0.1)',
  hover: '0px 10px 20px rgba(0, 0, 0, 0.2)',
};

const gradients = {
  darkGreenLightGreen: 'linear-gradient(#0B301C, #BBDED7)',
  lightGreenWhite: 'linear-gradient(#BBDED7, #FCFCFC)',
  darkGreenWhite: 'linear-gradient(#0B301C, #FCFCFC)',
  mediumGrayLightGreen: 'linear-gradient(#595959, #BBDED7)',
  darkGrayWhite: 'linear-gradient(#262626, #FCFCFC)',
  mediumGrayLightRed: 'linear-gradient(#595959, #DF8C87)',
};

const defaultTheme = {
  colors,
  gradients,
  device,
  id,
  fontWeights,
  fontSizes,
  size,
  shadows,
};

export default defaultTheme;
