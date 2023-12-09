// trace_skeleton.c
// Trace skeletonization result into polylines
//
// Lingdong Huang 2020
//
// dependencies:
// libpng, X11 (Optional)
// compile:
// gcc trace_skeleton.c -O3 -I /opt/X11/include -L/opt/X11/lib -lX11 -lpng -lm -lpthread -std=c99
// use:
// ./a.out path/to/image.png > output.txt

//================================
// PARAMS
//================================
#define VIEW_SCALE 3            // scale the displayed result
#define CHUNK_SIZE 10           // the chunk size
#define SHOW_GUI 1              // display result with gui
#define SAVE_RECTS 1            // additionally save bounding rects of chunks (for visualization)
#define TEST_SPEED 1000         // run a speed comparison between sequential & parallel
#define MAX_ITER 999            // maximum number of iterations

//================================
// INCLUDES
//================================

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#if SHOW_GUI
  #include <X11/Xlib.h>
  #include <X11/Xutil.h>
  #include <X11/Xos.h>
#endif

#include <png.h>

//================================
// ENUMS
//================================
#define HORIZONTAL 1
#define VERTICAL 2

//================================
// GLOBALS
//================================
typedef unsigned char uchar;
uchar* im; // the image
int W;     // width
int H;     // height

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

//================================
// DATASTRUCTURES
//================================

typedef struct _point_t {
  int x;
  int y;
  struct _point_t * next;
} point_t;

typedef struct _polyline_t {
  point_t* head;
  point_t* tail;
  struct _polyline_t* prev;
  struct _polyline_t* next;
  int size;
} polyline_t;

#if SAVE_RECTS
  typedef struct _rect_t {
    int x;
    int y;
    int w;
    int h;
    struct _rect_t* next;
  } rect_t;

  struct _rects_t{
    rect_t* head;
    rect_t* tail;
  } rects;
#endif

//================================
// DATASTRUCTURE IMPLEMENTATION
//================================

polyline_t* new_polyline(){
  polyline_t* q0 = (polyline_t*)malloc(sizeof(polyline_t));
  q0->head = NULL;
  q0->tail = NULL;
  q0->prev = NULL;
  q0->next = NULL;
  q0->size = 0;
  return q0;
}
void print_polyline(polyline_t* q){
  if (!q){
    return;
  }
  point_t* jt = q->head;
  while(jt){
    printf("%d,%d ",jt->x,jt->y);
    jt = jt->next;
  }
  printf("\n");
}
void print_polylines(polyline_t* q){
  if (!q){
    return;
  }
  polyline_t* it = q;
  while(it){
    point_t* jt = it->head;
    while(jt){
      printf("%d,%d ",jt->x,jt->y);
      jt = jt->next;
    }
    printf("\n");
    it = it->next;
  }
}
void destroy_polylines(polyline_t* q){
  if (!q){
    return;
  }
  polyline_t* it = q;
  while(it){
    polyline_t* lt = it->next;
    point_t* jt = it->head;
    while(jt){
      point_t* kt = jt->next;
      free(jt);
      jt = kt;
    }
    free(it);
    it = lt;
  }
}

void reverse_polyline(polyline_t* q){
  if (!q || (q->size < 2)){
    return;
  }
  q->tail->next = q->head;
  point_t* it0 = q->head;
  point_t* it1 = it0->next;
  point_t* it2 = it1->next;
  int i; for (i = 0; i < q->size-1; i++){
      it1->next = it0;
      it0 = it1;
      it1 = it2;
      it2 = it2->next;
  }
  point_t* q_head = q->head;
  q->head = q->tail;
  q->tail = q_head;
  q->tail->next = NULL;
}

void cat_tail_polyline(polyline_t* q0, polyline_t* q1){
  if (!q1){
    return;
  }
  if (!q0){
    *q0 = *new_polyline();
  }
  if (!q0->head){
    q0->head = q1->head;
    q0->tail = q1->tail;
    return;
  }
  q0->tail->next = q1->head;
  q0->tail = q1->tail;
  q0->size += q1->size;
  q0->tail->next = NULL;
}

void cat_head_polyline(polyline_t* q0, polyline_t* q1){
  if (!q1){
    return;
  }
  if (!q0){
    *q0 = *new_polyline();
  }
  if (!q1->head){
    return;
  }
  if (!q0->head){
    q0->head = q1->head;
    q0->tail = q1->tail;
    return;
  }
  q1->tail->next = q0->head;
  q0->head = q1->head;
  q0->size += q1->size;
  q0->tail->next = NULL;
}

void add_point_to_polyline(polyline_t* q, int x, int y){
  point_t* p = (point_t*)malloc(sizeof(point_t));
  p->x = x;
  p->y = y;
  p->next = NULL;
  if (!q->head){
    q->head = p;
    q->tail = p;
  }else{
    q->tail->next = p;
    q->tail = p;
  }
  q->size++;
}

polyline_t* prepend_polyline(polyline_t* q0, polyline_t* q1){
  if (!q0){
    return q1;
  }
  q1->next = q0;
  q0->prev = q1;
  return q1;
}

#if SAVE_RECTS
  void print_rects(){
    rect_t* it = rects.head;
    while(it){
      printf("%d %d %d %d\n",it->x,it->y,it->w,it->h);
      it = it->next;
    }
  }
  void destroy_rects(){
    rect_t* it = rects.head;
    while(it){
      rect_t* jt = it->next;
      free(it);
      it = jt;
    }
    rects.head = NULL;
    rects.tail = NULL;
  }
  void add_rect(int x, int y, int w, int h){
    rect_t* r = (rect_t*)malloc(sizeof(rect_t));
    r->x = x;
    r->y = y;
    r->w = w;
    r->h = h;
    r->next = NULL;
    if (!rects.head){
      rects.head = r;
      rects.tail = r;
    }else{
      rects.tail->next = r;
      rects.tail = r;
    }
  }
#endif

//================================
// RASTER SKELETONIZATION
//================================
// Binary image thinning (skeletonization) in-place.
// Implements Zhang-Suen algorithm.
// http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
int thinning_zs_iteration(int iter) {
  int diff = 0;
  for (int i = 1; i < H-1; i++){
    for (int j = 1; j < W-1; j++){
      int p2 = im[(i-1)*W+j]   & 1;
      int p3 = im[(i-1)*W+j+1] & 1;
      int p4 = im[(i)*W+j+1]   & 1;
      int p5 = im[(i+1)*W+j+1] & 1;
      int p6 = im[(i+1)*W+j]   & 1;
      int p7 = im[(i+1)*W+j-1] & 1;
      int p8 = im[(i)*W+j-1]   & 1;
      int p9 = im[(i-1)*W+j-1] & 1;
      
      int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
      int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
      int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
      int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
      if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
        im[i*W+j] |= 2;
    }
  }
  for (int i = 0; i < H*W; i++){
    int marker = im[i]>>1;
    int old = im[i]&1;
    im[i] = old & (!marker);
    if ((!diff) && (im[i] != old)){
      diff = 1;
    }
  }
  return diff;
};
void thinning_zs(){
  int diff = 1;
  do {
    diff &= thinning_zs_iteration(0);
    diff &= thinning_zs_iteration(1);
  }while (diff);
}

//================================
// MAIN ALGORITHM
//================================

// check if a region has any white pixel
int not_empty(int x, int y, int w, int h){
  for (int i = y; i < y+h; i++){
    for (int j = x; j < x+w; j++){
      if (im[i*W+j]){
        return 1;
      }
    }
  }
  return 0;
}

/**merge ith fragment of second chunk to first chunk
 * @param c0   fragments from  first  chunk
 * @param c1i  ith fragment of second chunk
 * @param sx   (x or y) coordinate of the seam
 * @param isv  is vertical, not horizontal?
 * @param mode 2-bit flag, 
 *             MSB = is matching the left (not right) end of the fragment from first  chunk
 *             LSB = is matching the right (not left) end of the fragment from second chunk
 * @return     matching successful?             
 */
int merge_impl(polyline_t* c0, polyline_t* c1i, int sx, int isv, int mode){
  int b0 = (mode >> 1 & 1)>0; // match c0 left
  int b1 = (mode >> 0 & 1)>0; // match c1 left
  polyline_t* c0j = NULL;
  int md = 4; // maximum offset to be regarded as continuous

  point_t* p1 = b1 ? c1i->head : c1i->tail;

  if (abs((isv?(p1->y):(p1->x))-sx)>0){ // not on the seam, skip
    return 0;
  }
  // find the best match
  polyline_t* it = c0;
  while (it){
    point_t* p0 = b0?(it->head):(it->tail);
    if (abs((isv?(p0->y):(p0->x))-sx)>1){ // not on the seam, skip
      it = it->next;
      continue;
    }
    int d = abs((isv?(p0->x):(p0->y)) - (isv?(p1->x):(p1->y)));
    if (d < md){
      c0j = it;
      md = d;
    }
    it = it->next;
  }

  if (c0j){ // best match is good enough, merge them
    if (b0 && b1){
      reverse_polyline(c1i);
      cat_head_polyline(c0j,c1i);
    }else if (!b0 && b1){
      cat_tail_polyline(c0j,c1i);
    }else if (b0 && !b1){
      cat_head_polyline(c0j,c1i);
    }else {
      reverse_polyline(c1i);
      cat_tail_polyline(c0j,c1i);
    }
    return 1;    
  }
  return 0;
}

/**merge fragments from two chunks
 * @param c0   fragments from first  chunk
 * @param c1   fragments from second chunk
 * @param sx   (x or y) coordinate of the seam
 * @param dr   merge direction, HORIZONTAL or VERTICAL?
 */
polyline_t* merge_frags(polyline_t* c0, polyline_t* c1, int sx, int dr){
  if (!c0){
    return c1;
  }
  if (!c1){
    return c0;
  }
  polyline_t* it = c1;
  while(it){
    polyline_t* tmp = it->next;
    if (dr == HORIZONTAL){
      if (merge_impl(c0,it,sx,0,1))goto rem;
      if (merge_impl(c0,it,sx,0,3))goto rem;
      if (merge_impl(c0,it,sx,0,0))goto rem;
      if (merge_impl(c0,it,sx,0,2))goto rem;
    }else{
      if (merge_impl(c0,it,sx,1,1))goto rem;
      if (merge_impl(c0,it,sx,1,3))goto rem;
      if (merge_impl(c0,it,sx,1,0))goto rem;
      if (merge_impl(c0,it,sx,1,2))goto rem;      
    }
    goto next;
    rem:
    if (!it->prev){
      c1 = it->next;
      if (it->next){
        it->next->prev = NULL;
      }
    }else{
      it->prev->next = it->next;
      if (it->next){
        it->next->prev = it->prev;
      }
    }
    free(it);
    next:
    it = tmp;
  }
  it = c1;
  while(it){
    polyline_t* tmp = it->next;
    it->prev = NULL;
    it->next = NULL;
    c0 = prepend_polyline(c0,it);
    it = tmp;
  }
  return c0;
}

/**recursive bottom: turn chunk into polyline fragments;
 * look around on 4 edges of the chunk, and identify the "outgoing" pixels;
 * add segments connecting these pixels to center of chunk;
 * apply heuristics to adjust center of chunk
 *
 * @param x    left of   chunk
 * @param y    top of    chunk
 * @param w    width of  chunk
 * @param h    height of chunk
 * @return     the polyline fragments
 */
polyline_t* chunk_to_frags(int x, int y, int w, int h){
  polyline_t* frags = NULL;
  int fsize = 0;
  int on = 0; // to deal with strokes thicker than 1px
  int li=-1, lj=-1;
  
  // walk around the edge clockwise
  for (int k = 0; k < h+h+w+w-4; k++){
    int i, j;
    if (k < w){
      i = y+0; j = x+k;
    }else if (k < w+h-1){
      i = y+k-w+1; j = x+w-1;
    }else if (k < w+h+w-2){
      i = y+h-1; j = x+w-(k-w-h+3); 
    }else{
      i = y+h-(k-w-h-w+4); j = x+0;
    }
    if (im[i*W+j]){ // found an outgoing pixel
      if (!on){     // left side of stroke
        on = 1;
        polyline_t* f = new_polyline();
        add_point_to_polyline(f, j, i);
        add_point_to_polyline(f, x+w/2,y+h/2);
        frags = prepend_polyline(frags,f);
        fsize ++;
      }
    }else{
      if (on){// right side of stroke, average to get center of stroke
        frags->head->x = (frags->head->x+lj)/2;
        frags->head->y = (frags->head->y+li)/2;
        on = 0;
      }
    }
    li = i;
    lj = j;
  }
  if (fsize == 2){ // probably just a line, connect them
    polyline_t* f = new_polyline();
    add_point_to_polyline(f,frags->head->x,frags->head->y);
    add_point_to_polyline(f,frags->next->head->x,frags->next->head->y);
    destroy_polylines(frags);
    frags = f;
  }else if (fsize > 2){ // it's a crossroad, guess the intersection
    int ms = 0;
    int mi = -1;
    int mj = -1;
    // use convolution to find brightest blob
    for (int i = y+1; i < y+h-1; i++){
      for (int j = x+1; j < x+w-1; j++){
        int s = 
          (im[i*W-W+j-1]) + (im[i*W-W+j]) + (im[i*W-W+j-1+1])+
          (im[i*W+j-1]  ) +   (im[i*W+j]) +   (im[i*W+j+1]  )+
          (im[i*W+W+j-1]) + (im[i*W+W+j]) + (im[i*W+W+j+1]  );
        if (s > ms){
          mi = i;
          mj = j;
          ms = s;
        }else if (s == ms && abs(j-(x+w/2))+abs(i-(y+h/2)) < abs(mj-(x+w/2))+abs(mi-(y+h/2))){
          mi = i;
          mj = j;
          ms = s;
        }
      }
    }
    if (mi != -1){
      polyline_t* it = frags;
      while(it){
        it->tail->x = mj;
        it->tail->y = mi;
        it = it->next;
      }
    }
  }
  return frags;
}



/**Trace skeleton from thinning result.
 * Algorithm:
 * 1. if chunk size is small enough, reach recursive bottom and turn it into segments
 * 2. attempt to split the chunk into 2 smaller chunks, either horizontall or vertically;
 *    find the best "seam" to carve along, and avoid possible degenerate cases
 * 3. recurse on each chunk, and merge their segments
 *
 * @param x       left of   chunk
 * @param y       top of    chunk
 * @param w       width of  chunk
 * @param h       height of chunk
 * @param iter    current iteration
 * @return        an array of polylines
*/
polyline_t* trace_skeleton(x,y,w,h,iter){

  polyline_t* frags = NULL;
  
  if (iter >= MAX_ITER){ // gameover
    return frags;
  }
  if (w <= CHUNK_SIZE && h <= CHUNK_SIZE){ // recursive bottom
    frags = chunk_to_frags(x,y,w,h);
    return frags;
  }
 

  int ms = INT_MAX; // number of white pixels on the seam, less the better
  int mi = -1; // horizontal seam candidate
  int mj = -1; // vertical   seam candidate
  
  if (h > CHUNK_SIZE){ // try splitting top and bottom
    for (int i = y+3; i < y+h-3; i++){
      if (im[i*W+x] ||im[(i-1)*W+x] ||im[i*W+x+w-1] ||im[(i-1)*W+x+w-1]){
        continue;
      }
      int s = 0;
      for (int j = x; j < x+w; j++){
        s += im[i*W+j];
        s += im[(i-1)*W+j];
      }
      if (s < ms){
        ms = s; mi = i;
      }else if (s == ms && abs(i-(y+h/2))<abs(mi-(y+h/2))){
        // if there is a draw (very common), we want the seam to be near the middle
        // to balance the divide and conquer tree
        ms = s; mi = i;
      }
    }
  }
  

  if (w > CHUNK_SIZE){ // same as above, try splitting left and right
    for (int j = x+3; j < x+w-3; j++){
      if (im[W*y+j]||im[W*(y+h)-W+j]||im[W*y+j-1]||im[W*(y+h)-W+j-1]){
        continue;
      }
      int s = 0;
      for (int i = y; i < y+h; i++){
        s += im[i*W+j]?1:0;
        s += im[i*W+j-1]?1:0;
      }
      if (s < ms){
        ms = s;
        mi = -1; // horizontal seam is defeated
        mj = j;
      }else if (s == ms && abs(j-(x+w/2))<abs(mj-(x+w/2))){
        ms = s;
        mi = -1;
        mj = j;
      }
    }
  }

  int L0=-1; int L1; int L2; int L3;
  int R0=-1; int R1; int R2; int R3;
  int dr = 0;
  int sx;
  if (h > CHUNK_SIZE && mi != -1){ // split top and bottom
    L0 = x; L1 = y;  L2 = w; L3 = mi-y;
    R0 = x; R1 = mi; R2 = w; R3 = y+h-mi;
    dr = VERTICAL;
    sx = mi;
  }else if (w > CHUNK_SIZE && mj != -1){ // split left and right
    L0 = x; L1 = y; L2 = mj-x; L3 = h;
    R0 = mj;R1 = y; R2 =x+w-mj;R3 = h;
    dr = HORIZONTAL;
    sx = mj;
  }
  if (dr!=0 && not_empty(L0,L1,L2,L3)){ // if there are no white pixels, don't waste time
    #if SAVE_RECTS
      add_rect(L0,L1,L2,L3);
    #endif
    frags = trace_skeleton(L0,L1,L2,L3,iter+1);
  }
  if (dr!=0 && not_empty(R0,R1,R2,R3)){
    #if SAVE_RECTS
      add_rect(R0,R1,R2,R3);
    #endif
    frags = merge_frags(frags, trace_skeleton(R0,R1,R2,R3,iter+1),sx,dr);
  }
  if (mi == -1 && mj == -1){ // splitting failed! do the recursive bottom instead
    frags = chunk_to_frags(x,y,w,h);
  }
  return frags;
}


//================================
// GUI/IO
//================================

// use libpng to read png into array of 0s and 1s
int read_png_as_bitmap(char* file_name){
  // based on http://zarb.org/~gc/html/libpng.html
  int x, y;
  int width, height;
  png_byte color_type;
  png_byte bit_depth;
  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep * row_pointers;
  char header[8];
  FILE *fp = fopen(file_name, "rb");
  if (!fp)return -1;
  fread(header, 1, 8, fp);
  if (png_sig_cmp((void*)header, 0, 8))exit(1);
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)return -1;
  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)return -1;
  if (setjmp(png_jmpbuf(png_ptr)))return -1;
  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);
  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);
  if (setjmp(png_jmpbuf(png_ptr)))return -1;
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  if (!row_pointers)return -1;
  for (y=0; y<height; y++)
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));
  png_read_image(png_ptr, row_pointers);
  int channels;
  if (color_type == PNG_COLOR_TYPE_RGB){
    channels = 3;
  }else if (color_type == PNG_COLOR_TYPE_RGBA){
    channels = 4;
  }else if (color_type == PNG_COLOR_TYPE_GRAY){
    channels = 1;
  }else{
    return -1;
  }
  W = width; H = height;
  im = (uchar*) malloc(sizeof(uchar)*W*H);
  if (!im)return -1;
  for (y=0; y<H; y++) {
    png_byte* row = row_pointers[y];
    for (x=0; x<W; x++) {
      png_byte* ptr = &(row[x*channels]);
      im[y*W+x] = (uchar)(ptr[0]>128?1:0);
    }
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_info_struct(png_ptr,&info_ptr);
  png_destroy_read_struct(&png_ptr,(png_infop*)NULL,(png_infop*)NULL);
  fclose(fp);
  return 0;
}

void print_bitmap(){
  for (int i = 0; i < H; i++){
    for (int j = 0; j < W; j++){
      printf("%d",im[i*W+j]);
    }
    printf("\n");
  }
}


#if SHOW_GUI

  Display *dis;
  int screen;
  Window win;
  GC gc;

  void init_x11(){    
    unsigned long black,white;
    dis=XOpenDisplay((char *)0);
    screen=DefaultScreen(dis);
    black=BlackPixel(dis, screen),
    white=WhitePixel(dis, screen);
    win=XCreateSimpleWindow(dis,DefaultRootWindow(dis),0,0, 
        W*VIEW_SCALE,H*VIEW_SCALE, 5,black, white);
    XSetStandardProperties(dis,win,"trace_skeleton.c","trace_skeleton.c",None,NULL,0,NULL);
    XSelectInput(dis, win, ExposureMask|ButtonPressMask|KeyPressMask);
        gc=XCreateGC(dis, win, 0,0);        
    XSetBackground(dis,gc,white);
    XSetForeground(dis,gc,black);
    XClearWindow(dis, win);
    XMapRaised(dis, win);
  }

  char* make_x11_bitmap(){
    int ww = ceil((float)(W*VIEW_SCALE)/8.0)*8;
    char* bmp = (char*) calloc(ceil(((float)(ww*H*VIEW_SCALE)/8.0)),sizeof(char));
    
    for (int i = 0; i < W*H; i++){
      int x = i % W;
      int y = i / W;
      
      if (!im[i]){
        int ii = (y*VIEW_SCALE)*ww+(x*VIEW_SCALE);
        int b = ii%8;
        int n = ii/8;
        bmp[n] = bmp[n]|(1<<(b));
      }
    }
    return bmp;
  }
#endif

int main(int argc, char** argv){
  #if SAVE_RECTS
    rects.head = NULL;
    rects.tail = NULL;
  #endif

  if (read_png_as_bitmap(argv[1])<0){
    printf("Error reading PNG. Abort.");exit(1);
  }
  #if SHOW_GUI
    char* bmp;
    bmp = make_x11_bitmap();
  #endif

  print_bitmap();
  thinning_zs();
  print_bitmap();

  polyline_t* p = NULL;

  if (TEST_SPEED){
    clock_t t; 
    t = clock();
    int n = TEST_SPEED;
    for (int i = 0; i < n; i++){
      destroy_polylines(p);
      #if SAVE_RECTS
        destroy_rects();
      #endif
      p = trace_skeleton(0,0,W,H,0);
    }
    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("time taken: %f s / %d runs\n",time_taken,n);
    destroy_polylines(p);
    #if SAVE_RECTS
      destroy_rects();
    #endif
  }
  
  p = trace_skeleton(0,0,W,H,0);
  print_polylines(p);

  #if SHOW_GUI

    //================================
    // DRAW RESULT WITH X11
    //================================

    XEvent event;
    KeySym key;
    char text[255];
    init_x11();

    Pixmap pix = XCreateBitmapFromData(dis, win, (char*)bmp, W*VIEW_SCALE,H*VIEW_SCALE);

    while(1) { 
      XNextEvent(dis, &event);
      if (event.type==Expose && event.xexpose.count==0) {
        XClearWindow(dis, win);

        XSetForeground(dis,gc,0);
        XCopyPlane(dis, pix, win, gc,0, 0,W*VIEW_SCALE, H*VIEW_SCALE,0, 0,1);
        
        #if SAVE_RECTS
          rect_t* rt = rects.head;

          XSetForeground(dis,gc,0x444444);
          XSetLineAttributes(dis, gc, 1, LineSolid, CapRound, JoinMiter);
          while (rt){
            XDrawRectangle(dis,win,gc,rt->x*VIEW_SCALE,rt->y*VIEW_SCALE,rt->w*VIEW_SCALE,rt->h*VIEW_SCALE);
            rt = rt->next;
          }
        #endif

        polyline_t* it = p;

        XSetLineAttributes(dis, gc, 2, LineSolid, CapRound, JoinMiter);

        while(it){
          
          point_t* jt = it->head;

          XSetForeground(dis,gc,rand()%0xFFFFFF);

          while(jt){
            point_t* kt = jt->next;
            XDrawRectangle(dis,win,gc,jt->x*VIEW_SCALE-1,jt->y*VIEW_SCALE-1,2,2);

            if (kt){
              XDrawLine(dis,win,gc,jt->x*VIEW_SCALE,jt->y*VIEW_SCALE,kt->x*VIEW_SCALE,kt->y*VIEW_SCALE);
            }
            jt = kt;
          }
          it = it->next;
        }

      }
    }

  #endif /*SHOW_GUI*/
  
  destroy_polylines(p);
  #if SAVE_RECTS
    destroy_rects();
  #endif
  free(im);
  return 0;
}