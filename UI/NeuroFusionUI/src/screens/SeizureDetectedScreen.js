import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  ScrollView,
  Animated,
  Vibration,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import NotificationService from '../services/NotificationService';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';

const SeizureDetectedScreen = ({ onGoBack, confidence, numSamples, timeTaken, fileName }) => {
  const [showBanner, setShowBanner] = useState(false);
  const bannerOpacity = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(-100)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    triggerAlert();
    saveToHistory();
  }, []);

  //ALERT LOGIC

  const triggerAlert = () => {
    Vibration.vibrate([0, 200, 100, 200, 100, 200]);

    setShowBanner(true);

    Animated.parallel([
      Animated.spring(slideAnim, {
        toValue: 0,
        friction: 6,
        tension: 40,
        useNativeDriver: true,
      }),
      Animated.timing(bannerOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();

    const pulseLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.07,
          duration: 700,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 700,
          useNativeDriver: true,
        }),
      ])
    );

    pulseLoop.start();

    NotificationService.sendSeizureAlert();

    const timer = setTimeout(() => {
      Animated.timing(bannerOpacity, {
        toValue: 0,
        duration: 400,
        useNativeDriver: true,
      }).start(() => setShowBanner(false));
      pulseLoop.stop();
    }, 5000);

    return () => {
      clearTimeout(timer);
      Vibration.cancel();
      pulseLoop.stop();
    };
  };

  // SAVE TO HISTORY 

  const saveToHistory = async () => {
    try {
      const existing = await AsyncStorage.getItem('analysisHistory');
      const history = existing ? JSON.parse(existing) : [];

      const newEntry = {
        id: Date.now().toString(),
        fileName,
        confidence,
        numSamples,
        timeTaken,
        date: new Date().toLocaleString(),
        result: 'Seizure Detected',
      };

      const updated = [newEntry, ...history];

      await AsyncStorage.setItem(
        'analysisHistory',
        JSON.stringify(updated)
      );
    } catch (error) {
      console.log('Error saving history:', error);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#E63946" />

      {/* In-App Banner */}
      {showBanner && (
        <Animated.View
          style={[
            styles.bannerContainer,
            {
              opacity: bannerOpacity,
              transform: [
                { translateY: slideAnim },
                { scale: pulseAnim },
              ],
            },
          ]}
        >
          <LinearGradient
            colors={['#FF6B6B', '#E63946']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.banner}
          >
            {/* Animated bell icon */}
            <Animated.View
              style={[
                styles.iconContainer,
                {
                  transform: [
                    {
                      rotate: pulseAnim.interpolate({
                        inputRange: [1, 1.05],
                        outputRange: ['-10deg', '10deg'],
                      }),
                    },
                  ],
                },
              ]}
            >
              <MaterialCommunityIcons name="bell-alert" size={28} color="#FFFFFF" />
            </Animated.View>

            {/* Alert content */}
            <View style={styles.bannerContent}>
              <Text style={styles.bannerTitle}>SEIZURE DETECTED!</Text>
              <Text style={styles.bannerSubtitle}>Take necessary precautions</Text>
            </View>

            {/* Close button */}
            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setShowBanner(false)}
            >
              <MaterialCommunityIcons name="close" size={20} color="#FFFFFF" />
            </TouchableOpacity>
          </LinearGradient>
        </Animated.View>
      )}
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.content}>
          {/* Warning Icon */}
          <View style={styles.iconContainer}>
            <MaterialCommunityIcons name="alert" size={80} color="#1A1A1A" />
          </View>

          {/* Title */}
          <Text style={styles.title}>Seizure Detected</Text>
          <Text style={styles.confidenceScore}>Confidence: {confidence}%</Text>

          {/* Summary Box */}
          <View style={styles.summaryBox}>
            <Text style={styles.summaryTitle}>Summary</Text>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Total Time Taken:</Text>
              <Text style={styles.summaryValue}>{timeTaken}s</Text>
            </View>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Number of events checked:</Text>
              <Text style={styles.summaryValue}>{numSamples}</Text>
            </View>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>File Name:</Text>
              <Text style={styles.summaryValue}>{fileName}</Text>
            </View>
          </View>

          {/* Warning Message */}
          <Text style={styles.warning}>
            Immediate medical attention recommended. Please contact your
            healthcare provider.
          </Text>

          {/* Button */}
          <TouchableOpacity
            style={styles.button}
            onPress={onGoBack}
          >
            <Text style={styles.buttonText}>Go back to dashboard</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#E63946',
  },
  scrollContent: {
    flexGrow: 1,
  },
  content: {
    flex: 1,
    padding: 24,
    justifyContent: 'center',
  },
  iconContainer: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 4,
    borderColor: '#1A1A1A',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
  },
  confidenceScore: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginTop: 12,
    marginBottom: 32,
  },
  summaryBox: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
  },
  summaryTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1A1A1A',
    marginBottom: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  summaryLabel: {
    fontSize: 16,
    color: '#1A1A1A',
    flex: 1,
  },
  summaryValue: {
    fontSize: 16,
    color: '#1A1A1A',
    fontWeight: '600',
  },
  warning: {
    fontSize: 16,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 16,
    fontWeight: '600',
  },
  button: {
    backgroundColor: '#B844FF',
    borderRadius: 28,
    padding: 16,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  bannerContainer: {
    position: 'absolute',
    top: 20,
    left: 16,
    right: 16,
    zIndex: 999,
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderRadius: 16,
    shadowColor: '#E63946',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  bannerContent: {
    flex: 1,
  },
  bannerTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
    letterSpacing: 0.5,
    marginBottom: 2,
  },
  bannerSubtitle: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 13,
    fontWeight: '500',
  },
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
});

export default SeizureDetectedScreen;