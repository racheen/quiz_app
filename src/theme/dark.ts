/* eslint-disable no-magic-numbers */
import { rem } from 'polished';

const id = 2;

const colors = {
  white: '#FFFFFF', // White text and highlights
  lightGray: '#A5A5A5', // Light Gray for subtle text or accents
  lightGreen: '#588157', // Light Green for highlights in dark theme
  darkGreen: '#344E41', // Dark Green for the main secondary color
  darkGray: '#595959', // Dark Gray for background
  mediumGray: '#2A2A2A', // Medium Gray for some elements or section separation
  lightRed: '#720714', // Light Red for warning or highlights
  primary: '#397367', // Primary Light Green Color
  secondary: '#063A21', // Secondary Dark Green Color
  text: '#DAD7CD', // White text on dark background
  accent: '#3A5A40', // Event Card Background or Accent Color
  background: '#595959', // Main Background Color
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
  dropShadowCard: '1px 2px 4px #BBDED7', // Light Green for shadow effect
  card: '0px 4px 6px rgba(0, 0, 0, 0.3)', // Darker shadow for a more intense look
  hover: '0px 10px 20px rgba(0, 0, 0, 0.4)', // Hover effect with stronger shadow
};

const gradients = {
  darkGreenLightGreen: 'linear-gradient(#0B301C, #BBDED7)', // Dark Green to Light Green
  lightGreenWhite: 'linear-gradient(#BBDED7, #FCFCFC)',
  darkGreenWhite: 'linear-gradient(#0B301C, #FFFFFF)', // Dark Green to White
  mediumGrayLightGreen: 'linear-gradient(#2A2A2A, #BBDED7)', // Medium Gray to Light Green
  darkGrayWhite: 'linear-gradient(#1D1D1D, #FFFFFF)', // Dark Gray to White
  mediumGrayLightRed: 'linear-gradient(#595959, #DF8C87)',
};

const darkTheme = {
  colors,
  gradients,
  device,
  id,
  fontWeights,
  fontSizes,
  size,
  shadows,
};

export default darkTheme;
