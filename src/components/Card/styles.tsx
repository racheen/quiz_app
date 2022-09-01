import styled, { css } from 'styled-components';

type CardProps = {
  isProject?: boolean;
};

export const Container = styled.div<CardProps>`
  display: flex;
  width: 200px;
  background: ${(props) => props.theme.colors.lightGray};
  box-shadow: ${(props) => props.theme.shadows.dropShadowCard};
  flex-direction: column;
  margin: 20px 5px;

  @media screen and ${(props) => props.theme.device.tablet} {
    width: 300px;
  }

  ${(props) =>
    props.isProject &&
    css`
      width: 200px;

      @media screen and ${props.theme.device.tablet} {
        width: 400px;
      } ;
    `}

  a {
    text-decoration: none;
    display: flex;
    flex-direction: column;
  }

  :hover {
    filter: drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25));
    cursor: pointer;
  }
`;

export const CardContentLayout = styled.div`
  padding: 21px 10px 6px 20px;
  color: ${(props) => props.theme.colors.darkGreen};
  font-weight: ${(props) => props.theme.fontWeights.regular};

  @media screen and ${(props) => props.theme.device.tablet} {
    padding: 43px 12px 12px 30px;
    font-size: ${(props) => props.theme.fontSizes.large};
  } ;
`;

export const Title = styled(CardContentLayout)<CardProps>`
  font-weight: ${(props) => props.theme.fontWeights.semiBold};
  font-size: ${(props) => props.theme.fontSizes.medium};
  max-height: 100px;
  min-height: 50px;

  ${(props) =>
    props.isProject &&
    css`
      height: 10px;
    `}

  @media screen and ${(props) => props.theme.device.tablet} {
    font-size: ${(props) => props.theme.fontSizes.large};
    height: 60px;
  } ;
`;

export const MediumGrayText = styled.span`
  color: ${(props) => props.theme.colors.mediumGray};
`;

export const LightGreenText = styled.span`
  color: ${(props) => props.theme.colors.lightGreen};
`;

export const Content = styled(CardContentLayout)`
  padding: 5px 10px 6px 20px;
  font-size: ${(props) => props.theme.fontSizes.normal};
  color: ${(props) => props.theme.colors.mediumGray};
  max-height: 150px;
  min-height: 100px;

  @media screen and ${(props) => props.theme.device.tablet} {
    padding: 10px 40px 25px 30px;
    font-size: ${(props) => props.theme.fontSizes.medium};
    height: 150px;
  } ;
`;

export const Subcontent = styled(CardContentLayout)`
  font-size: ${(props) => props.theme.fontSizes.small};
  font-weight: ${(props) => props.theme.fontWeights.light};
  padding-bottom: 30px;

  @media screen and ${(props) => props.theme.device.tablet} {
    font-weight: ${(props) => props.theme.fontWeights.regular};
    padding-bottom: 40px;
  } ;
`;
