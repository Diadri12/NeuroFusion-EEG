import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, StatusBar, SafeAreaView,
  ScrollView, Animated, Alert,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
 
// Helpers
// Cleans blob-URL filenames (e.g. "21e632e0-…-ef773ae.csv") → "EEG File"
const cleanFileName = (name) => {
  if (!name || name === 'unknown') return 'EEG File';
  // UUID pattern: 8-4-4-4-12 hex chars
  const uuidRe = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;
  const base   = name.replace(/\.csv$/i, '').replace(/\.txt$/i, '');
  if (uuidRe.test(base)) return 'EEG Recording';
  return name.length > 28 ? name.slice(0, 26) + '…' : name;
};
 
// Mini progress bar
const MiniBar = ({ pct, color }) => {
  const w = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(w, { toValue: pct / 100, duration: 700, useNativeDriver: false }).start();
  }, []);
  return (
    <View style={miniStyles.bg}>
      <Animated.View style={[miniStyles.fill, {
        width: w.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] }),
        backgroundColor: color,
      }]} />
    </View>
  );
};
const miniStyles = StyleSheet.create({
  bg:   { flex: 1, height: 5, backgroundColor: '#F0F0F0', borderRadius: 4, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 4 },
});
 
// Single history card
const HistoryCard = ({ item, onDelete, slideAnim, fadeAnim }) => {
  const [expanded,   setExpanded]  = useState(false);
  const expandAnim  = useRef(new Animated.Value(0)).current;
  const chevronAnim = useRef(new Animated.Value(0)).current;
 
  const isSeizure   = item.urgency === 'critical' || item.result === 'Seizure Detected';
  const accent      = isSeizure ? '#E74C3C' : '#1A8A4A';
  const accentLight = isSeizure ? '#FEECEB' : '#E8F8EE';
  const icon        = isSeizure ? 'alert-circle' : 'check-circle';
  const label       = isSeizure ? 'Seizure' : 'Clear';
 
  const toggleExpand = () => {
    const v = expanded ? 0 : 1;
    setExpanded(!expanded);
    Animated.parallel([
      Animated.spring(expandAnim,  { toValue: v, friction: 9, tension: 65, useNativeDriver: false }),
      Animated.timing(chevronAnim, { toValue: v, duration: 180,            useNativeDriver: true  }),
    ]).start();
  };
 
  const maxHeight     = expandAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 240] });
  const chevronRotate = chevronAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '180deg'] });
 
  const confirmDelete = () =>
    Alert.alert('Delete Record', 'Remove this entry from history?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => onDelete(item.id) },
    ]);
 
  // Parse date into friendly parts
  const [datePart, timePart] = (item.date || '').split(', ');
 
  return (
    <Animated.View style={[
      styles.card,
      { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
    ]}>
      {/* ── Main tappable row ── */}
      <TouchableOpacity
        style={styles.cardMain}
        onPress={toggleExpand}
        activeOpacity={0.75}
      >
        {/* Left: status icon */}
        <View style={[styles.resultIcon, { backgroundColor: accentLight }]}>
          <MaterialCommunityIcons name={icon} size={22} color={accent} />
        </View>
 
        {/* Centre: filename + date */}
        <View style={styles.cardCentre}>
          <View style={styles.cardTopLine}>
            <View style={[styles.pill, { backgroundColor: accentLight }]}>
              <Text style={[styles.pillText, { color: accent }]}>{label}</Text>
            </View>
            <Text style={styles.cardName} numberOfLines={1}>
              {cleanFileName(item.fileName)}
            </Text>
          </View>
          <Text style={styles.cardDate}>
            {datePart}
            {timePart ? <Text style={styles.cardTime}>  {timePart}</Text> : null}
          </Text>
        </View>
 
        {/* Right: chevron */}
        <Animated.View style={{ transform: [{ rotate: chevronRotate }] }}>
          <MaterialCommunityIcons name="chevron-down" size={20} color="#C0C0C0" />
        </Animated.View>
      </TouchableOpacity>
 
      {/* ── Meta chips row ── */}
      <View style={styles.metaRow}>
        <View style={styles.chip}>
          <MaterialCommunityIcons name="clock-outline" size={12} color="#999" />
          <Text style={styles.chipText}>{item.timeTaken ?? '—'}s</Text>
        </View>
        <View style={styles.chip}>
          <MaterialCommunityIcons name="pulse" size={12} color="#999" />
          <Text style={styles.chipText}>{item.totalWindows ?? '—'} windows</Text>
        </View>
        <TouchableOpacity style={styles.deleteChip} onPress={confirmDelete}>
          <MaterialCommunityIcons name="trash-can-outline" size={13} color="#E74C3C" />
          <Text style={styles.deleteChipText}>Delete</Text>
        </TouchableOpacity>
      </View>
 
      {/* ── Expandable distribution section ── */}
      <Animated.View style={[styles.expandWrap, { maxHeight }]}>
        <View style={[styles.expandInner, { borderTopColor: '#F0F0F0' }]}>
          <Text style={styles.expandTitle}>Class Distribution</Text>
          {[
            { label: 'Interictal', pct: item.interictalPct ?? 0, color: '#27AE60' },
            { label: 'Preictal',   pct: item.preictalPct   ?? 0, color: '#F39C12' },
            { label: 'Ictal',      pct: item.ictalPct      ?? 0, color: '#E74C3C' },
          ].map(row => (
            <View key={row.label} style={styles.distRow}>
              <Text style={styles.distLabel}>{row.label}</Text>
              <MiniBar pct={row.pct} color={row.color} />
              <Text style={styles.distPct}>{row.pct}%</Text>
            </View>
          ))}
          {item.advice ? (
            <View style={[styles.adviceBox, { borderLeftColor: accent }]}>
              <Text style={styles.adviceText}>{item.advice}</Text>
            </View>
          ) : null}
        </View>
      </Animated.View>
    </Animated.View>
  );
};
 
// Main Screen
const HistoryScreen = () => {
  const router = useRouter();
  const [searchQuery,  setSearchQuery]  = useState('');
  const [historyData,  setHistoryData]  = useState([]);
  const [filter,       setFilter]       = useState('all'); // 'all' | 'seizure' | 'clear'
  const headerFade = useRef(new Animated.Value(0)).current;
  const cardAnims  = useRef([]).current; // per-card { fade, slide }
 
  const loadHistory = async () => {
    try {
      const stored = await AsyncStorage.getItem('analysisHistory');
      const data   = stored ? JSON.parse(stored) : [];
      setHistoryData(data);
    } catch (e) { console.log('Load history error:', e); }
  };
 
  // Reload every time screen is focused
  useFocusEffect(useCallback(() => {
    loadHistory();
    Animated.timing(headerFade, { toValue: 1, duration: 400, useNativeDriver: true }).start();
  }, []));
 
  // Build per-card animations when data changes
  useEffect(() => {
    cardAnims.length = 0;
    historyData.forEach((_, i) => {
      const fade  = new Animated.Value(0);
      const slide = new Animated.Value(24);
      cardAnims.push({ fade, slide });
      Animated.parallel([
        Animated.timing(fade,  { toValue: 1, duration: 350, delay: i * 70, useNativeDriver: true }),
        Animated.spring(slide, { toValue: 0, friction: 8, tension: 50, delay: i * 70, useNativeDriver: true }),
      ]).start();
    });
  }, [historyData]);
 
  const deleteItem = async (id) => {
    const updated = historyData.filter(h => h.id !== id);
    setHistoryData(updated);
    await AsyncStorage.setItem('analysisHistory', JSON.stringify(updated));
  };
 
  const clearAll = () =>
    Alert.alert('Clear History', 'Delete all analysis records?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear All', style: 'destructive',
        onPress: async () => {
          setHistoryData([]);
          await AsyncStorage.removeItem('analysisHistory');
        },
      },
    ]);
 
  const filtered = historyData.filter(item => {
    const matchSearch =
      (item.fileName || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.result   || '').toLowerCase().includes(searchQuery.toLowerCase());
    const matchFilter =
      filter === 'all'     ? true :
      filter === 'seizure' ? item.urgency === 'critical' || item.result === 'Seizure Detected' :
                             item.urgency !== 'critical' && item.result !== 'Seizure Detected';
    return matchSearch && matchFilter;
  });
 
  const seizureCount = historyData.filter(
    h => h.urgency === 'critical' || h.result === 'Seizure Detected'
  ).length;
  const clearCount = historyData.length - seizureCount;
 
  return (
    <SafeAreaView style={styles.container}>
      {/* <StatusBar barStyle="dark-content" backgroundColor="#FFF" /> */}
 
      {/* ── Header ── */}
      <Animated.View style={[styles.header, { opacity: headerFade }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Analysis History</Text>
        {historyData.length > 0 ? (
          <TouchableOpacity onPress={clearAll} style={styles.clearBtn}>
            <MaterialCommunityIcons name="delete-sweep" size={22} color="#E74C3C" />
          </TouchableOpacity>
        ) : <View style={{ width: 36 }} />}
      </Animated.View>
 
      {/* ── Summary chips ── */}
      <Animated.View style={[styles.summaryRow, { opacity: headerFade }]}>
        <View style={styles.summaryChip}>
          <MaterialCommunityIcons name="history" size={15} color="#B844FF" />
          <Text style={styles.summaryChipText}>{historyData.length} total</Text>
        </View>
        <View style={[styles.summaryChip, { backgroundColor: '#FFF5F5' }]}>
          <MaterialCommunityIcons name="alert-circle" size={15} color="#E74C3C" />
          <Text style={[styles.summaryChipText, { color: '#E74C3C' }]}>{seizureCount} seizures</Text>
        </View>
        <View style={[styles.summaryChip, { backgroundColor: '#F0FFF5' }]}>
          <MaterialCommunityIcons name="check-circle" size={15} color="#1A8A4A" />
          <Text style={[styles.summaryChipText, { color: '#1A8A4A' }]}>{clearCount} clear</Text>
        </View>
      </Animated.View>
 
      {/* ── Search bar ── */}
      <View style={styles.searchWrap}>
        <MaterialCommunityIcons name="magnify" size={19} color="#AAA" />
        <TextInput
          style={styles.searchInput}
          placeholder="Search by file name or result..."
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholderTextColor="#BBB"
        />
        {searchQuery.length > 0 && (
          <TouchableOpacity onPress={() => setSearchQuery('')}>
            <MaterialCommunityIcons name="close-circle" size={18} color="#CCC" />
          </TouchableOpacity>
        )}
      </View>
 
      {/* ── Filter tabs ── */}
      <View style={styles.filterRow}>
        {[
          { key: 'all',     label: 'All',     icon: 'history'       },
          { key: 'seizure', label: 'Seizure',  icon: 'alert-circle'  },
          { key: 'clear',   label: 'Clear',    icon: 'check-circle'  },
        ].map(tab => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.filterTab, filter === tab.key && styles.filterTabActive]}
            onPress={() => setFilter(tab.key)}
          >
            <MaterialCommunityIcons
              name={tab.icon}
              size={14}
              color={filter === tab.key ? '#FFF' : '#888'}
            />
            <Text style={[styles.filterTabText, filter === tab.key && styles.filterTabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
 
      {/* ── List ── */}
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      >
        {filtered.length === 0 ? (
          <View style={styles.emptyWrap}>
            <MaterialCommunityIcons name="clipboard-text-off" size={56} color="#DDD" />
            <Text style={styles.emptyTitle}>
              {historyData.length === 0 ? 'No history yet' : 'No results found'}
            </Text>
            <Text style={styles.emptySub}>
              {historyData.length === 0
                ? 'Upload an EEG file to get started'
                : 'Try changing your search or filter'}
            </Text>
          </View>
        ) : (
          filtered.map((item, index) => {
            const anims = cardAnims[
              historyData.findIndex(h => h.id === item.id)
            ] || { fade: new Animated.Value(1), slide: new Animated.Value(0) };
            return (
              <HistoryCard
                key={item.id}
                item={item}
                index={index}
                onDelete={deleteItem}
                fadeAnim={anims.fade}
                slideAnim={anims.slide}
              />
            );
          })
        )}
      </ScrollView>
    </SafeAreaView>
  );
};
 
// Styles 
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F4F5F9' },
 
  // Header
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: '#FFF', paddingHorizontal: 18, paddingVertical: 14,
    borderBottomWidth: 1, borderBottomColor: '#EFEFEF',
  },
  backBtn: {
    width: 36, height: 36, borderRadius: 18,
    backgroundColor: '#F5F5F5', justifyContent: 'center', alignItems: 'center',
  },
  headerTitle: { fontSize: 20, fontWeight: '800', color: '#B844FF' },
  clearBtn: {
    width: 36, height: 36, borderRadius: 18,
    backgroundColor: '#FFF0F0', justifyContent: 'center', alignItems: 'center',
  },
 
  // Summary chips
  summaryRow: {
    flexDirection: 'row', gap: 8, paddingHorizontal: 18, paddingVertical: 12,
    backgroundColor: '#FFF', borderBottomWidth: 1, borderBottomColor: '#EFEFEF',
  },
  summaryChip: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#F5F0FF', borderRadius: 20,
    paddingHorizontal: 12, paddingVertical: 6,
  },
  summaryChipText: { fontSize: 12, fontWeight: '700', color: '#B844FF' },
 
  // Search
  searchWrap: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: '#FFF', borderRadius: 12,
    marginHorizontal: 18, marginTop: 14,
    paddingHorizontal: 14, paddingVertical: 10,
    borderWidth: 1, borderColor: '#E8E8E8',
  },
  searchInput: { flex: 1, fontSize: 14, color: '#333' },
 
  // Filter tabs
  filterRow: {
    flexDirection: 'row', gap: 8,
    paddingHorizontal: 18, marginTop: 12, marginBottom: 4,
  },
  filterTab: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 5, paddingVertical: 8, borderRadius: 20,
    backgroundColor: '#FFF', borderWidth: 1, borderColor: '#E0E0E0',
  },
  filterTabActive:     { backgroundColor: '#B844FF', borderColor: '#B844FF' },
  filterTabText:       { fontSize: 12, fontWeight: '600', color: '#888' },
  filterTabTextActive: { color: '#FFF' },
 
  // List
  scroll:      { flex: 1 },
  listContent: { paddingHorizontal: 18, paddingTop: 12, paddingBottom: 100 },
 
  // Card
  card: {
    backgroundColor: '#FFF', borderRadius: 16, marginBottom: 10,
    overflow: 'hidden',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07, shadowRadius: 8, elevation: 3,
  },
 
  // Main tappable row
  cardMain: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 14, paddingTop: 14, paddingBottom: 10, gap: 12,
  },
  resultIcon: {
    width: 44, height: 44, borderRadius: 22,
    justifyContent: 'center', alignItems: 'center',
    flexShrink: 0,
  },
  cardCentre:  { flex: 1, gap: 4 },
  cardTopLine: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  pill: {
    borderRadius: 6, paddingHorizontal: 7, paddingVertical: 2,
  },
  pillText: { fontSize: 10, fontWeight: '800', letterSpacing: 0.4 },
  cardName: { fontSize: 14, fontWeight: '700', color: '#1A1A2E', flex: 1 },
  cardDate: { fontSize: 11, color: '#AAAAAA' },
  cardTime: { fontSize: 11, color: '#C8C8C8' },
 
  // Meta chips row
  metaRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: 14, paddingBottom: 12,
  },
  chip: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: '#F6F6F6', borderRadius: 8,
    paddingHorizontal: 9, paddingVertical: 5,
  },
  chipText:   { fontSize: 11, color: '#888', fontWeight: '500' },
  deleteChip: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: '#FFF0F0', borderRadius: 8,
    paddingHorizontal: 9, paddingVertical: 5,
    marginLeft: 'auto',
  },
  deleteChipText: { fontSize: 11, color: '#E74C3C', fontWeight: '600' },
 
  // Expanded section
  expandWrap:  { overflow: 'hidden' },
  expandInner: {
    borderTopWidth: 1, marginHorizontal: 14,
    paddingTop: 12, paddingBottom: 14,
  },
  expandTitle: {
    fontSize: 10, fontWeight: '800', color: '#BBBBBB',
    letterSpacing: 1, textTransform: 'uppercase', marginBottom: 10,
  },
  distRow:   { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 9 },
  distLabel: { width: 68, fontSize: 12, color: '#666', fontWeight: '600' },
  distPct:   { width: 34, fontSize: 12, color: '#444', fontWeight: '700', textAlign: 'right' },
 
  adviceBox: {
    marginTop: 10, borderLeftWidth: 3,
    backgroundColor: '#FAFAFA', borderRadius: 6,
    padding: 10,
  },
  adviceText: { fontSize: 12, color: '#666', lineHeight: 18 },
 
  // Empty
  emptyWrap:  { alignItems: 'center', paddingTop: 80, gap: 12 },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: '#CCC' },
  emptySub:   { fontSize: 13, color: '#CCC', textAlign: 'center' },
});
 
export default HistoryScreen;