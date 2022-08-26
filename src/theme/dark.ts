/* eslint-disable no-magic-numbers */
import { rem } from 'polished';

const id = 2;
const colors = {
  white: '#fcfcfc',
  lightGray: '#f9f9f9',
  lightGreen: '#bbded7',
  lightRed: '#DF8C87',
  darkGray: '#262626',
  darkGreen: '#0b301c',
  mediumGray: '#595959',
};

const fontWeights = {
  bold: 700,
  light: 300,
  medium: 500,
  regular: 400,
  semiBold: 600,
};

const fontSizes = {
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
  mobileL: '426px',
  mobileM: '376px',
  mobileS: '320px',
  tablet: '768px',
};

const device = {
  desktop: `(min-width: ${size.desktop})`,
  desktopL: `(min-width: ${size.desktop})`,
  laptop: `(min-width: ${size.laptop})`,
  laptopL: `(min-width: ${size.laptopL})`,
  mobileL: `(min-width: ${size.mobileL})`,
  mobileM: `(min-width: ${size.mobileM})`,
  mobileS: `(min-width: ${size.mobileS})`,
  tablet: `(min-width: ${size.tablet})`,
};

const shadows = {
  dropShadowCard: '1px 2px 4px #BBDED7',
};

const darkTheme = {
  colors,
  device,
  id,
  fontWeights,
  fontSizes,
  size,
  shadows,
};

export default darkTheme;
