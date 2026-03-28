import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  Animated,
  Dimensions,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

const SplashScreen = () => {
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.3)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const loaderAnim = useRef(new Animated.Value(0)).current;
  const textSlideAnim = useRef(new Animated.Value(50)).current;
  const textFadeAnim = useRef(new Animated.Value(0)).current;
  const particleAnims = useRef([
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
  ]).current;

  useEffect(() => {
    // Sequential animations
    Animated.sequence([
      // 1. Fade in and scale up the brain icon
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.spring(scaleAnim, {
          toValue: 1,
          friction: 4,
          tension: 50,
          useNativeDriver: true,
        }),
      ]),
      // 2. Small delay
      Animated.delay(200),
      // 3. Slide up and fade in text
      Animated.parallel([
        Animated.timing(textSlideAnim, {
          toValue: 0,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(textFadeAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ]),
    ]).start();

    // Continuous pulsing animation for the brain
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.15,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();

    // Continuous rotation for loader
    Animated.loop(
      Animated.timing(rotateAnim, {
        toValue: 1,
        duration: 2000,
        useNativeDriver: true,
      })
    ).start();

    // Loader progress animation
    Animated.timing(loaderAnim, {
      toValue: 1,
      duration: 2500,
      useNativeDriver: false,
    }).start();

    // Particle animations
    particleAnims.forEach((anim, index) => {
      Animated.loop(
        Animated.sequence([
          Animated.delay(index * 200),
          Animated.timing(anim, {
            toValue: 1,
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0,
            duration: 0,
            useNativeDriver: true,
          }),
        ])
      ).start();
    });
  }, []);

  const spin = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const loaderWidth = loaderAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.container}>
      {/* <StatusBar barStyle="light-content" backgroundColor="#B844FF" /> */}
      
      {/* Animated particles/dots in background */}
      <View style={styles.particlesContainer}>
        {particleAnims.map((anim, index) => {
          const translateY = anim.interpolate({
            inputRange: [0, 1],
            outputRange: [0, -200],
          });
          const opacity = anim.interpolate({
            inputRange: [0, 0.5, 1],
            outputRange: [0, 1, 0],
          });
          
          return (
            <Animated.View
              key={index}
              style={[
                styles.particle,
                {
                  left: `${(index + 1) * 18}%`,
                  opacity,
                  transform: [{ translateY }],
                },
              ]}
            />
          );
        })}
      </View>

      {/* Main content */}
      <Animated.View
        style={[
          styles.content,
          {
            opacity: fadeAnim,
            transform: [{ scale: scaleAnim }],
          },
        ]}
      >
        {/* Pulsing brain icon container */}
        <Animated.View
          style={[
            styles.iconContainer,
            {
              transform: [{ scale: pulseAnim }],
            },
          ]}
        >
          {/* Rotating circle background */}
          <Animated.View
            style={[
              styles.rotatingCircle,
              {
                transform: [{ rotate: spin }],
              },
            ]}
          >
            <View style={styles.rotatingCircleSegment} />
          </Animated.View>

          {/* Brain icon */}
          <View style={styles.brainIconWrapper}>
            <MaterialCommunityIcons name="brain" size={100} color="#FFFFFF" />
          </View>
        </Animated.View>

        {/* App title with slide animation */}
        <Animated.View
          style={{
            opacity: textFadeAnim,
            transform: [{ translateY: textSlideAnim }],
          }}
        >
          <Text style={styles.title}>EpiGuard</Text>
          <Text style={styles.subtitle}>EEG Seizure Detection</Text>
        </Animated.View>
      </Animated.View>

      {/* Progress loader at bottom */}
      <Animated.View style={styles.loaderContainer}>
        <View style={styles.loaderBackground}>
          <Animated.View
            style={[
              styles.loaderFill,
              {
                width: loaderWidth,
              },
            ]}
          />
        </View>
        <Animated.Text
          style={[
            styles.loaderText,
            { opacity: textFadeAnim },
          ]}
        >
          Loading...
        </Animated.Text>
      </Animated.View>

      {/* Spinning loader indicator */}
      <Animated.View
        style={[
          styles.spinnerContainer,
          {
            transform: [{ rotate: spin }],
            opacity: textFadeAnim,
          },
        ]}
      >
        <View style={styles.spinner}>
          <View style={styles.spinnerSegment} />
        </View>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#B844FF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  particlesContainer: {
    ...StyleSheet.absoluteFillObject,
    overflow: 'hidden',
  },
  particle: {
    position: 'absolute',
    bottom: 0,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
  },
  content: {
    alignItems: 'center',
    marginBottom: 100,
  },
  iconContainer: {
    position: 'relative',
    width: 180,
    height: 180,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30,
  },
  rotatingCircle: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 90,
    borderWidth: 3,
    borderColor: 'transparent',
    borderTopColor: 'rgba(255, 255, 255, 0.4)',
    borderRightColor: 'rgba(255, 255, 255, 0.4)',
  },
  rotatingCircleSegment: {
    width: '100%',
    height: '100%',
  },
  brainIconWrapper: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  title: {
    fontFamily: 'RobotoSlab_600SemiBold',
    fontSize: 36,
    color: '#FFFFFF',
    textAlign: 'center',
  },
  subtitle: {
    fontFamily: 'RobotoSlab_300Light',
    fontSize: 16,
    color: '#FFFFFF',
    opacity: 0.9,
    textAlign: 'center',
    letterSpacing: 1,
  },
  loaderContainer: {
    position: 'absolute',
    bottom: 80,
    width: width * 0.7,
    alignItems: 'center',
  },
  loaderBackground: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  loaderFill: {
    height: '100%',
    backgroundColor: '#FFFFFF',
    borderRadius: 2,
  },
  loaderText: {
    color: '#FFFFFF',
    fontSize: 14,
    marginTop: 12,
    fontWeight: '500',
    letterSpacing: 1,
  },
  spinnerContainer: {
    position: 'absolute',
    bottom: 140,
  },
  spinner: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    borderTopColor: '#FFFFFF',
  },
});

export default SplashScreen;