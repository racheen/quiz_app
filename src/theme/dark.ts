const id = 2;
const colors = {
  primary: '#66dbb1',
  onPrimary: '#003828',
  primaryContainer: '#00513c',
  onPrimaryContainer: '#84f8cd',
  primaryFixed: '#84f8cd',
  onPrimaryFixed: '#002116',
  primaryFixedDim: '#66dbb1',
  onPrimaryFixedVariant: '#00513c',
  secondary: '#b3ccbf',
  onSecondary: '#1e352c',
  secondaryContainer: '#354c42',
  onSecondaryContainer: '#cee9db',
  secondaryFixed: '#cee9db',
  onSecondaryFixed: '#082017',
  secondaryFixedDim: '#b3ccbf',
  onSecondaryFixedVariant: '#354c42',
  tertiary: '#a6cce0',
  onTertiary: '#0a3445',
  tertiaryContainer: '#264b5c',
  onTertiaryContainer: '#c2e8fd',
  tertiaryFixed: '#c2e8fd',
  onTertiaryFixed: '#001f2a',
  tertiaryFixedDim: '#a6cce0',
  onTertiaryFixedVariant: '#264b5c',
  error: '#ffb4ab',
  errorContainer: '#93000a',
  onError: '#690005',
  onErrorContainer: '#ffdad6',
  background: '#191c1a',
  onBackground: '#e1e3e0',
  outline: '#89938d',
  inverseOnSurface: '#191c1a',
  inverseSurface: '#e1e3e0',
  inversePrimary: '#006c50',
  shadow: '#000000',
  surfaceTint: '#66dbb1',
  outlineVariant: '#404944',
  scrim: '#000000',
  surface: '#111412',
  onSurface: '#c5c7c4',
  surfaceVariant: '#404944',
  onSurfaceVariant: '#bfc9c2',
  surfaceContainerHighest: '#323633',
  surfaceContainerHigh: '#272b29',
  surfaceContainer: '#1d201e',
  surfaceContainerLow: '#191c1a',
  surfaceContainerLowest: '#0c0f0d',
  surfaceDim: '#111412',
  surfaceBright: '#373a38',
};

const android = {
  colorAccentPrimary: '#84f8cd',
  colorAccentPrimaryVariant: '#47bf97',
  colorAccentSecondary: '#cee9db',
  colorAccentSecondaryVariant: '#98b1a4',
  colorAccentTertiary: '#c2e8fd',
  colorAccentTertiaryVariant: '#8cb0c4',
  textColorPrimary: '#eff1ee',
  textColorSecondary: '#bfc9c2',
  textColorTertiary: '#89938d',
  textColorPrimaryInverse: '#191c1a',
  textColorSecondaryInverse: '#444845',
  textColorTertiaryInverse: '#757875',
  colorBackground: '#191c1a',
  colorBackgroundFloating: '#191c1a',
  colorSurface: '#2e312f',
  colorSurfaceVariant: '#444845',
  colorSurfaceHighlight: '#505351',
  surfaceHeader: '#444845',
  underSurface: '#000000',
  offState: '#2e312f',
  accentSurface: '#e9f3ec',
  textPrimaryOnAccent: '#191c1a',
  textSecondaryOnAccent: '#404944',
  volumeBackground: '#393c3a',
  scrim: '#c5c7c4',
};

const fonts = {
  body: {
    large: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Regular',
      fontWeight: 400,
      fontSize: 16,
      letterSpacing: 0.5,
    },
    medium: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Regular',
      fontWeight: 400,
      fontSize: 14,
      letterSpacing: 0.25,
    },
    small: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Regular',
      fontWeight: 400,
      fontSize: 12,
      letterSpacing: 0.4000000059604645,
    },
  },
  label: {
    large: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Medium',
      fontWeight: 500,
      fontSize: 14,
      lineHeight: 20,
      letterSpacing: 0.10000000149011612,
    },
    medium: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Medium',
      fontWeight: 500,
      fontSize: 12,
      lineHeight: 16,
      letterSpacing: 0.5,
    },
    small: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Medium',
      fontWeight: 500,
      fontSize: 11,
      lineHeight: 16,
      letterSpacing: 0.5,
    },
  },
  title: {
    large: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Regular',
      fontWeight: 400,
      fontSize: 22,
      lineHeight: 28,
      letterSpacing: 0,
    },
    medium: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Medium',
      fontWeight: 500,
      fontSize: 16,
      lineHeight: 24,
      letterSpacing: 0.15000000596046448,
    },
    small: {
      fontFamilyName: 'Roboto',
      fontFamilyStyle: 'Medium',
      fontWeight: 500,
      fontSize: 14,
      lineHeight: 20,
      letterSpacing: 0.10000000149011612,
    },
  },
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
  android,
  colors,
  device,
  fonts,
  id,
  size,
  shadows,
};

export default darkTheme;
