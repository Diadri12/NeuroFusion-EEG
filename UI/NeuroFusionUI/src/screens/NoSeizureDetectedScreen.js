import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  ScrollView,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const NoSeizureDetectedScreen = ({ onGoBack, confidence, numSamples, timeTaken, fileName }) => {

  useEffect(() => {
      saveToHistory();
    }, []);

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
        result: 'No Seizure Detected',
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
      <StatusBar barStyle="light-content" backgroundColor="#00D66A" />
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.content}>
          {/* Check Icon */}
          <View style={styles.iconContainer}>
            <MaterialCommunityIcons name="check" size={80} color="#00D66A" />
          </View>

          {/* Title */}
          <Text style={styles.title}>No Seizure Detected</Text>
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

          {/* Disclaimer */}
          <Text style={styles.disclaimer}>
            Disclaimer: This is not a medical diagnosis. Always consult your
            doctor if you have any health concern.
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
    backgroundColor: '#00D66A',
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
    backgroundColor: '#FFFFFF',
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
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
  },
  summaryTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  summaryLabel: {
    fontSize: 16,
    color: '#FFFFFF',
    flex: 1,
  },
  summaryValue: {
    fontSize: 16,
    color: '#FFFFFF',
    fontWeight: '600',
  },
  disclaimer: {
    fontSize: 14,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 16,
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
});

export default NoSeizureDetectedScreen;