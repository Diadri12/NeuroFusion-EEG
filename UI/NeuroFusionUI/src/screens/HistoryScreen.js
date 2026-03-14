import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  ScrollView,
  Animated,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useIsFocused } from '@react-navigation/native';

const HistoryScreen = () => {
  const router = useRouter();
  const isFocused = useIsFocused();
  const [searchQuery, setSearchQuery] = useState('');
  const [historyData, setHistoryData] = useState([]);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnimsRef = useRef([]);

  // Load history from AsyncStorage
  const loadHistory = async () => {
    try {
      const stored = await AsyncStorage.getItem('analysisHistory');
      const data = stored ? JSON.parse(stored) : [];
      setHistoryData(data);

      // Create slide animations based on loaded data
      slideAnimsRef.current = data.map(() => new Animated.Value(30));
    } catch (error) {
      console.log('Error loading history:', error);
    }
  };

    // Reload history whenever screen is focused
  useEffect(() => {
    if (isFocused) loadHistory();
  }, [isFocused]);

  // Animate cards whenever historyData changes
  useEffect(() => {
    if (!slideAnimsRef.current.length) return;

    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      ...slideAnimsRef.current.map((anim, index) =>
        Animated.spring(anim, {
          toValue: 0,
          delay: index * 100,
          friction: 8,
          tension: 40,
          useNativeDriver: true,
        })
      ),
    ]).start();
  }, [historyData]);

  const filteredData = historyData.filter(
    (item) =>
      item.fileName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.result.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleGoBack = () => router.back();

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />

      {/* Header */}
      <Animated.View style={[styles.header, { opacity: fadeAnim }]}>
        <View style={styles.statusBar}>
          <Text style={styles.time}>10:00</Text>
          <View style={styles.statusIcons}>
            <MaterialCommunityIcons name="signal" size={16} color="#000" />
            <MaterialCommunityIcons name="wifi" size={16} color="#000" style={{ marginLeft: 4 }} />
            <MaterialCommunityIcons name="battery" size={16} color="#000" style={{ marginLeft: 4 }} />
          </View>
        </View>

        <TouchableOpacity style={styles.backButton} onPress={handleGoBack}>
          <MaterialCommunityIcons name="arrow-left" size={24} color="#000" />
        </TouchableOpacity>

        <Text style={styles.headerTitle}>Analysis History</Text>
      </Animated.View>

      {/* Search Bar */}
      <Animated.View
        style={[
          styles.searchContainer,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnimsRef.current[0] || 0 }],
          },
        ]}
      >
        <MaterialCommunityIcons name="magnify" size={20} color="#999" style={styles.searchIcon} />
        <TextInput
          style={styles.searchInput}
          placeholder="Search by file name or result"
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholderTextColor="#999"
        />
      </Animated.View>

      {/* History List */}
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.historyContainer}
        showsVerticalScrollIndicator={false}
      >
        {filteredData.length === 0 ? (
          <Text style={{ textAlign: 'center', marginTop: 40, color: '#999' }}>
            No history found
          </Text>
        ) : (
          filteredData.map((item, index) => (
            <Animated.View
              key={item.id}
              style={[
                styles.historyCard,
                {
                  opacity: fadeAnim,
                  transform: [{ translateY: slideAnimsRef.current[Math.min(index, slideAnimsRef.current.length - 1)] || 0 }],
                },
              ]}
            >
              <View style={styles.cardContent}>
                <Text style={styles.cardLabel}>
                  Date: <Text style={styles.cardValue}>{item.date}</Text>
                </Text>
                <Text style={styles.cardLabel}>
                  File Name: <Text style={styles.cardValue}>{item.fileName}</Text>
                </Text>
                <Text style={styles.cardLabel}>
                  Result:{' '}
                  <Text style={[styles.cardValue, item.result === 'Seizure Detected' ? styles.seizureText : styles.noSeizureText]}>
                    {item.result}
                  </Text>
                </Text>
                <Text style={styles.cardLabel}>
                  Confidence: <Text style={styles.cardValue}>{item.confidence}%</Text>
                </Text>
                <Text style={styles.cardLabel}>
                  Duration: <Text style={styles.cardValue}>{item.timeTaken}s</Text>
                </Text>
              </View>
            </Animated.View>
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  header: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  time: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000',
  },
  statusIcons: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F5F5F5',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#B844FF',
    textAlign: 'center',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    marginHorizontal: 20,
    marginTop: 16,
    marginBottom: 16,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    paddingVertical: 12,
    fontSize: 16,
    color: '#333',
  },
  scrollView: {
    flex: 1,
  },
  historyContainer: {
    paddingHorizontal: 20,
    paddingBottom: 100,
  },
  historyCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardContent: {
    gap: 6,
  },
  cardLabel: {
    fontSize: 14,
    color: '#333',
    fontWeight: '600',
  },
  cardValue: {
    fontWeight: '400',
    color: '#666',
  },
  seizureText: {
    color: '#E63946',
    fontWeight: '600',
  },
  noSeizureText: {
    color: '#00D66A',
    fontWeight: '600',
  },
});

export default HistoryScreen;